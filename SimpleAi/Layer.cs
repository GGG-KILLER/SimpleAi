﻿using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using JetBrains.Annotations;
using SimpleAi.Math;

namespace SimpleAi;

/// <summary>
/// The base class for layers.
/// </summary>
/// <typeparam name="T">The numeric type used by this layer.</typeparam>
/// <remarks>
/// <para>Activation must be done on the layer itself.</para>
/// </remarks>
[PublicAPI]
public abstract class Layer<T> where T : unmanaged, INumber<T>
{
    protected readonly T[] MutBiases;
    protected readonly T[] MutWeights;

    /// <summary>
    /// Initializes a new instance of the layer.
    /// </summary>
    /// <param name="inputCount">The number of inputs this layer will accept.</param>
    /// <param name="outputCount">The number of outputs (neurons) this layer will have.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <list type="bullet">
    ///     <item>Thrown when <paramref name="inputCount"/> is negative or zero.</item>
    ///     <item>Thrown when <paramref name="outputCount"/> is negative or zero.</item>
    /// </list>
    /// </exception>
    protected Layer(int inputCount, int outputCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputCount);

        MutWeights = GC.AllocateUninitializedArray<T>(outputCount * inputCount);
        MutBiases  = GC.AllocateUninitializedArray<T>(outputCount);

        Inputs  = inputCount;
        Outputs = outputCount;
    }

    /// <summary>
    /// Number of inputs this layer accepts.
    /// </summary>
    [PublicAPI]
    public int Inputs { get; }

    /// <summary>
    /// Number of outputs this layer accepts.
    /// </summary>
    [PublicAPI]
    public int Outputs { get; }

    /// <summary>
    /// The weights used by this layer.
    /// </summary>
    [PublicAPI]
    public ReadOnlySpan<T> Weights => MutWeights.AsSpan();

    /// <summary>
    /// The biases used by this layer.
    /// </summary>
    [PublicAPI]
    public ReadOnlySpan<T> Biases => MutBiases.AsSpan();

    /// <summary>
    ///     <para>Randomizes the weights of this layer using a normal distribution.</para>
    ///     <para>Will generate weights that are <paramref name="mean"/> ± <paramref name="stdDev"/>.</para>
    /// </summary>
    /// <param name="mean">The middle point of the weight distribution.</param>
    /// <param name="stdDev">The margin of randomness to allow around the <paramref name="mean"/>.</param>
    public virtual void RandomizeWeights(T mean, T stdDev)
    {
        for (var nodeIdx = 0; nodeIdx < MutWeights.Length; nodeIdx++)
            MutWeights[nodeIdx] = MathEx.RandomBetweenNormalDistribution(Random.Shared, mean, stdDev);
    }

    /// <summary>
    /// Runs inference for this layer.
    /// </summary>
    /// <param name="inputs">The inputs to be processed by this layer.</param>
    /// <param name="outputs">The output buffer where the outputs of this layer will be stored.</param>
    /// <exception cref="ArgumentException">
    ///     <list type="bullet">
    ///         <item>Thrown when <paramref name="inputs"/> does not have a length of <b>at least</b> <see cref="Inputs"/>.</item>
    ///         <item>Thrown when <paramref name="outputs"/> does not have a length of <b>at least</b> <see cref="Outputs"/>.</item>
    ///     </list>
    /// </exception>
    [PublicAPI]
    public abstract void RunInference(ReadOnlySpan<T> inputs, Span<T> outputs);

    /// <summary>
    /// Calculates the cost gradients for this layer.
    /// </summary>
    /// <param name="averageCostFunction">The function to call to obtain the average cost of the network.</param>
    /// <param name="originalCost">The original cost before any modifications.</param>
    /// <param name="weightGradientCosts">The buffer where to store the cost gradients for the weights.</param>
    /// <param name="biasGradientCosts">The buffer where to store the cost gradients for the biases.</param>
    [SkipLocalsInit, PublicAPI]
    protected internal virtual void CalculateCostGradients(
        Func<T> averageCostFunction,
        T       originalCost,
        Span<T> weightGradientCosts,
        Span<T> biasGradientCosts)
    {
        Debug.Assert(weightGradientCosts.Length == MutWeights.Length);
        Debug.Assert(biasGradientCosts.Length == MutBiases.Length);

        T delta;
        if (typeof(T) == typeof(Half) || typeof(T) == typeof(double) || typeof(T) == typeof(float))
            delta = T.CreateChecked(value: 0.0001);
        else
            delta = T.CreateChecked(value: 1);

        ref T weightsStart = ref MutWeights.Ref();
        ref T weight       = ref weightsStart;
        ref T weightsEnd   = ref Unsafe.Add(ref weightsStart, MutWeights.Length);
        while (Unsafe.IsAddressLessThan(ref weight, ref weightsEnd))
        {
            weight += delta;
            T deltaCost = averageCostFunction() - originalCost;
            weight -= delta;
            weightGradientCosts.UnsafeIndex(
                (int) Unsafe.ByteOffset(ref weightsStart, ref weight) / Unsafe.SizeOf<T>()) = deltaCost / delta;

            weight = ref Unsafe.Add(ref weight, elementOffset: 1);
        }

        ref T biasesStart = ref MutBiases.Ref();
        ref T bias        = ref biasesStart;
        ref T biasesEnd   = ref Unsafe.Add(ref biasesStart, MutBiases.Length);
        while (Unsafe.IsAddressLessThan(ref bias, ref biasesEnd))
        {
            bias += delta;
            T deltaCost = averageCostFunction() - originalCost;
            bias -= delta;
            biasGradientCosts.UnsafeIndex((int) Unsafe.ByteOffset(ref biasesStart, ref bias) / Unsafe.SizeOf<T>()) =
                deltaCost / delta;

            bias = ref Unsafe.Add(ref bias, elementOffset: 1);
        }
    }

    /// <summary>
    /// Applies the cost gradients previously generated by <see cref="CalculateCostGradients"/> to the weights and biases.
    /// </summary>
    /// <param name="weightGradientCosts">The cost gradients to apply to the weights.</param>
    /// <param name="biasGradientCosts">The bias cost gradients to apply to the biases.</param>
    /// <param name="learnRate">The learn rate to apply the cost gradients with.</param>
    [SkipLocalsInit, PublicAPI]
    protected internal void ApplyCostGradients(Span<T> weightGradientCosts, Span<T> biasGradientCosts, T learnRate)
    {
        // weights = weightGradientCosts * learnRate
        MathEx.Binary<T, MulOp<T>>(weightGradientCosts, learnRate, weightGradientCosts);
        MathEx.Binary<T, SubOp<T>>(MutWeights, weightGradientCosts, MutWeights);
        weightGradientCosts.Clear();

        // biases -= biasGradientCosts * learnRate
        MathEx.Binary<T, MulOp<T>>(biasGradientCosts, learnRate, biasGradientCosts);
        MathEx.Binary<T, SubOp<T>>(MutBiases, biasGradientCosts, MutBiases);
        biasGradientCosts.Clear();
    }
}

/// <summary>
/// The actual implementation of the layers.
/// </summary>
/// <param name="inputCount">The number of inputs this layer will accept.</param>
/// <param name="outputCount">The number of outputs (neurons) this layer will have.</param>
/// <exception cref="ArgumentOutOfRangeException">
/// <list type="bullet">
///     <item>Thrown when <paramref name="inputCount"/> is negative or zero.</item>
///     <item>Thrown when <paramref name="outputCount"/> is negative or zero.</item>
/// </list>
/// </exception>
/// <typeparam name="T">The numeric type used by this layer.</typeparam>
/// <typeparam name="TActivation">The activation function type used by this layer.</typeparam>
[PublicAPI]
public sealed class Layer<T, TActivation>(int inputCount, int outputCount) : Layer<T>(inputCount, outputCount)
    where T : unmanaged, INumber<T> where TActivation : IActivationFunction<T>
{
    /// <inheritdoc />
    [SkipLocalsInit]
    public override void RunInference(ReadOnlySpan<T> inputs, Span<T> outputs)
    {
        // Localize to avoid multiple field reads
        int inputCount  = Inputs;
        int neuronCount = Outputs;

        if (inputs.Length != inputCount)
            throw new ArgumentException(
                $"Inputs are not the correct size. ({inputs.Length} != {inputCount})",
                nameof(inputs));
        if (outputs.Length < neuronCount)
            throw new ArgumentException(
                $"Outputs are not the correct size. ({outputs.Length} < {neuronCount})",
                nameof(outputs));

        ref T weights   = ref MutWeights.Ref();
        ref T output    = ref outputs.Ref();
        ref T outputEnd = ref outputs.UnsafeIndex(neuronCount);
        while (Unsafe.IsAddressLessThan(ref output, ref outputEnd))
        {
            output = MathEx.Aggregate<T, MulOp<T>, AddOp<T>>(
                MemoryMarshal.CreateReadOnlySpan(ref weights, Inputs),
                inputs);

            weights = ref Unsafe.Add(ref weights, Inputs);
            output  = ref Unsafe.Add(ref output, elementOffset: 1);
        }

        MathEx.Binary<T, AddOp<T>>(outputs, MutBiases, outputs);
        TActivation.Activate(outputs, outputs);
    }

    /// <summary>
    /// Loads a layer from its weights and biases without doing any validations.
    /// </summary>
    /// <param name="weights"></param>
    /// <param name="biases"></param>
    /// <returns>The loaded layer.</returns>
    [PublicAPI]
    public static Layer<T, TActivation> LoadUnsafe(T[] weights, T[] biases)
    {
        var layer = new Layer<T, TActivation>(weights.Length / biases.Length, biases.Length);

        weights.CopyTo(layer.MutWeights.AsSpan());
        biases.CopyTo(layer.MutBiases.AsSpan());

        return layer;
    }
}
