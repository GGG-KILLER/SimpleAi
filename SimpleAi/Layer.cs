using System.Numerics;
using System.Runtime.CompilerServices;
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
    protected readonly T[]        MutBiases;
    protected readonly Weights<T> MutWeights;

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

        MutWeights = new Weights<T>(GC.AllocateUninitializedArray<T>(outputCount * inputCount), inputCount);
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
    public ReadOnlyWeights<T> Weights => MutWeights;

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
    public virtual void RunInference(ReadOnlySpan<T> inputs, Span<T> outputs) => RunInferenceCore(inputs, outputs);

    /// <summary>
    /// Runs the inference process storing the activation inputs if provided storage for it.
    /// </summary>
    /// <param name="inputs">The inputs to run inference on.</param>
    /// <param name="outputs">The buffer to store inference results on.</param>
    /// <param name="activationInputs">
    /// The <paramref name="outputs"/> but without the activation function executed on them.
    /// </param>
    [PublicAPI]
    protected internal abstract void RunInferenceCore(
        ReadOnlySpan<T> inputs,
        Span<T>         outputs,
        Span<T>         activationInputs = default);

    /// <summary>
    /// Calculates the partial derivatives for the activation.
    /// </summary>
    [SkipLocalsInit, PublicAPI]
    protected internal abstract void CalculateActivationDerivatives(ReadOnlySpan<T> weightedInputs, Span<T> outputs);

    /// <summary>
    /// Applies the supplied cost gradients to the weights and biases.
    /// </summary>
    /// <param name="weightGradientCosts">The cost gradients to apply to the weights.</param>
    /// <param name="biasGradientCosts">The bias cost gradients to apply to the biases.</param>
    /// <param name="learnRate">The learn rate to apply the cost gradients with.</param>
    [SkipLocalsInit, PublicAPI]
    protected internal virtual void ApplyCostGradients(
        Span<T> weightGradientCosts,
        Span<T> biasGradientCosts,
        T       learnRate)
    {
        // weights -= weightGradientCosts * learnRate
        MathEx.Binary<T, MulOp<T>>(weightGradientCosts, learnRate, weightGradientCosts);
        MathEx.Binary<T, SubOp<T>>(MutWeights.AsSpan(), weightGradientCosts, MutWeights.AsSpan());
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
    protected internal override void RunInferenceCore(
        ReadOnlySpan<T> inputs,
        Span<T>         outputs,
        Span<T>         activationInputs = default)
    {
        if (inputs.Length != Inputs)
            throw new ArgumentException(
                $"Inputs are not the correct size. ({inputs.Length} != {Inputs})",
                nameof(inputs));

        // Output buffers can be larger, so we check if they can store at least the number of elements we need. 
        if (outputs.Length < Outputs)
            throw new ArgumentException(
                $"Outputs are not the correct size. ({outputs.Length} < {Outputs})",
                nameof(outputs));
        if (!activationInputs.IsEmpty && activationInputs.Length < Outputs)
            throw new ArgumentException(
                $"Activation inputs are not the correct size. ({activationInputs.Length} < {Outputs})",
                nameof(activationInputs));

        for (var nodeIdx = 0; nodeIdx < Outputs; nodeIdx++)
        {
            var weights = MutWeights.GetNodeWeights(nodeIdx);
            outputs.UnsafeIndex(nodeIdx) = MathEx.Aggregate<T, MulOp<T>, AddOp<T>>(weights, inputs);
        }

        MathEx.Binary<T, AddOp<T>>(outputs, MutBiases, outputs);
        if (!activationInputs.IsEmpty) outputs.CopyTo(activationInputs);
        TActivation.Activate(outputs, outputs);
    }

    /// <inheritdoc />
    protected internal override void CalculateActivationDerivatives(ReadOnlySpan<T> weightedInputs, Span<T> outputs)
        => TActivation.Derivative(weightedInputs, outputs);

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
