﻿using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SimpleAi.Math;

namespace SimpleAi;

public interface ILayer<T>
{
    int Inputs { get; }
    int Size { get; }

    // Init
    void RandomizeWeights(T mean, T stdDev);

    // Inference
    void RunInference(ReadOnlySpan<T> inputs, Span<T> outputs);

    // Training
    void CalculateCostGradients(
        TrainingSession<T> trainingSession,
        AverageCostFunc<T> getAverageCost,
        T originalCost);
    void ApplyCostGradients(TrainingSession<T> trainingSession, T learnRate);
}

public sealed class Layer<T, TActivation> : ILayer<T>
    where T : unmanaged, INumber<T>
    where TActivation : IActivationFunction<T>
{
    private readonly T[] _weights;
    private readonly T[] _biases;

    public int Inputs { get; }
    public int Size { get; }

    public Layer(int inputCount, int size)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(size);

        _weights = GC.AllocateUninitializedArray<T>(size * inputCount);
        _biases = GC.AllocateUninitializedArray<T>(size);

        Inputs = inputCount;
        Size = size;
    }

    public static Layer<T, TActivation> LoadUnsafe(T[] weights, T[] biases)
    {
        var layer = new Layer<T, TActivation>(weights.Length / biases.Length, biases.Length);

        weights.CopyTo(layer._weights.AsSpan());
        biases.CopyTo(layer._biases.AsSpan());

        return layer;
    }

    public void RandomizeWeights(T mean, T stdDev)
    {
        for (var nodeIdx = 0; nodeIdx < _weights.Length; nodeIdx++)
        {
            _weights[nodeIdx] = randomBetweenNormalDistribution(Random.Shared, mean, stdDev);
        }

        static T randomBetweenNormalDistribution(Random random, T mean, T stdDev)
        {
            var x1 = 1 - random.NextDouble();
            var x2 = 1 - random.NextDouble();

            double y1 = double.Sqrt(-2.0 * double.Log(x1)) * double.Cos(2.0 * double.Pi * x2);
            return T.CreateSaturating(y1 * double.CreateSaturating(stdDev) + double.CreateSaturating(mean));
        }
    }

    [SkipLocalsInit]
    public void RunInference(ReadOnlySpan<T> inputs, Span<T> outputs)
    {
        // Localize to avoid multiple field reads
        var inputCount = Inputs;
        var neuronCount = Size;

        if (inputs.Length != inputCount)
            throw new ArgumentException($"Inputs are not the correct size. ({inputs.Length} != {inputCount})", nameof(inputs));
        if (outputs.Length < neuronCount)
            throw new ArgumentException($"Outputs are not the correct size. ({outputs.Length} < {neuronCount})", nameof(outputs));

        var neuronIdx = 0;
        ref T weights = ref _weights!.Ref();
        ref T output = ref outputs.Ref();
        ref T outputEnd = ref outputs.UnsafeIndex(neuronCount);
        while (Unsafe.IsAddressLessThan(ref output, ref outputEnd))
        {
            output = MathEx.Aggregate<T, MulOp<T>, AddOp<T>>(
                MemoryMarshal.CreateReadOnlySpan(ref weights, Inputs),
                inputs);

            neuronIdx++;
            weights = ref Unsafe.Add(ref weights, Inputs);
            output = ref Unsafe.Add(ref output, 1);
        }

        MathEx.Binary<T, AddOp<T>>(outputs, _biases, outputs);
        TActivation.Activate(outputs, outputs);
    }

    [SkipLocalsInit]
    public void CalculateCostGradients(
        TrainingSession<T> trainingSession,
        AverageCostFunc<T> getAverageCost,
        T originalCost)
    {
        var layerData = trainingSession[this];
        var weightGradientCosts = layerData.WeightGradientCosts.Span;
        var biasGradientCosts = layerData.BiasGradientCosts.Span;

        T delta;
        if (typeof(T) == typeof(Half) || typeof(T) == typeof(double) || typeof(T) == typeof(float))
            delta = T.CreateChecked(0.0001);
        else
            delta = T.CreateChecked(1);

        ref T weightsStart = ref _weights.Ref();
        ref T weight = ref weightsStart;
        ref T weightsEnd = ref Unsafe.Add(ref weightsStart, _weights.Length);
        while (Unsafe.IsAddressLessThan(ref weight, ref weightsEnd))
        {
            weight += delta;
            var deltaCost = getAverageCost(trainingSession) - originalCost;
            weight -= delta;
            weightGradientCosts.UnsafeIndex((int)Unsafe.ByteOffset(ref weightsStart, ref weight) / Unsafe.SizeOf<T>()) = deltaCost / delta;

            weight = ref Unsafe.Add(ref weight, 1);
        }

        ref T biasesStart = ref _biases.Ref();
        ref T bias = ref biasesStart;
        ref T biasesEnd = ref Unsafe.Add(ref biasesStart, _biases.Length);
        while (Unsafe.IsAddressLessThan(ref bias, ref biasesEnd))
        {
            bias += delta;
            var deltaCost = getAverageCost(trainingSession) - originalCost;
            bias -= delta;
            biasGradientCosts.UnsafeIndex((int)Unsafe.ByteOffset(ref biasesStart, ref bias) / Unsafe.SizeOf<T>()) = deltaCost / delta;

            bias = ref Unsafe.Add(ref bias, 1);
        }
    }

    [SkipLocalsInit]
    public void ApplyCostGradients(TrainingSession<T> trainingSession, T learnRate)
    {
        var layerData = trainingSession[this];
        var weightGradientCosts = layerData.WeightGradientCosts.Span;
        var biasGradientCosts = layerData.BiasGradientCosts.Span;

        // weights = weightGradientCosts * learnRate
        MathEx.Binary<T, MulOp<T>>(weightGradientCosts, learnRate, _weights);
        weightGradientCosts.Clear();

        // biases -= biasGradientCosts * learnRate
        MathEx.Binary<T, MulOp<T>>(biasGradientCosts, learnRate, biasGradientCosts);
        MathEx.Binary<T, SubOp<T>>(_biases, biasGradientCosts, _biases);
        biasGradientCosts.Clear();
    }
}
