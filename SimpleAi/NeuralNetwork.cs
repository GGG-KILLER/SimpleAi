using System.Numerics;
using System.Runtime.CompilerServices;

namespace SimpleAi;

public interface INeuralNetwork<T>
{
    int Inputs { get; }

    int Outputs { get; }

    int LayerCount { get; }

    ILayer<T> this[Index index] { get; }

    // Initialize
    void RandomizeWeights(T mean, T stdDev);

    // Inference
    void RunInference(InferenceSession<T> inferenceSession, ReadOnlySpan<T> inputs, Span<T> output);

    // Training
    T AverageCost(TrainingSession<T> trainingSession);

    void Train(TrainingSession<T> trainingSession, T learnRate);
}

public sealed class NeuralNetwork<T, TActivation> : INeuralNetwork<T>
    where T : unmanaged, INumber<T> where TActivation : IActivationFunction<T>
{
    private readonly Layer<T, TActivation>[] _layers;

    public int Inputs { get; }

    public int Outputs { get; }

    public int LayerCount => _layers.Length;

    public ReadOnlySpan<Layer<T, TActivation>> Layers => _layers.AsSpan();

    public ILayer<T> this[Index index] => _layers[index];

    public NeuralNetwork(int inputs, params ReadOnlySpan<int> layerSizes)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputs);
        if (layerSizes.Length < 1)
        {
            throw new ArgumentException("At least one layer is required.", nameof(layerSizes));
        }

        Inputs  = inputs;
        Outputs = layerSizes[^1];
        _layers = new Layer<T, TActivation>[layerSizes.Length];

        for (var idx = 0; idx < layerSizes.Length; idx++)
        {
            if (layerSizes[idx] <= 0)
            {
                throw new ArgumentException("Layer sizes must be greater than zero.", nameof(layerSizes));
            }

            _layers[idx] = new Layer<T, TActivation>(idx == 0 ? inputs : layerSizes[idx - 1], layerSizes[idx]);
        }
    }

    private NeuralNetwork(Layer<T, TActivation>[] layers)
    {
        _layers = layers;
        Inputs  = layers[0].Inputs;
        Outputs = layers[^1].Size;
    }

    public static NeuralNetwork<T, TActivation> UnsafeLoad(T[][] weights, T[][] biases)
    {
        var layers = new Layer<T, TActivation>[weights.Length];
        for (var idx = 0; idx < weights.Length; idx++)
        {
            layers[idx] = Layer<T, TActivation>.LoadUnsafe(weights[idx], biases[idx]);
        }
        return new NeuralNetwork<T, TActivation>(layers);
    }

    public void RandomizeWeights(T mean, T stdDev)
    {
        for (int idx = 0; idx < _layers.Length; idx++)
        {
            _layers.UnsafeIndex(idx).RandomizeWeights(mean, stdDev);
        }
    }

    [SkipLocalsInit]
    public void RunInference(InferenceSession<T> inferenceSession, ReadOnlySpan<T> inputs, Span<T> output)
    {
        if (inputs.Length != Inputs)
        {
            throw new ArgumentException("Input vector does not have the correct amount of elements.", nameof(inputs));
        }
        if (output.Length != Outputs)
        {
            throw new ArgumentException("Output does not have the correct size.", nameof(output));
        }

        var inference = RunInferenceCore(inferenceSession, inputs);
        inference.CopyTo(output);
    }

    [SkipLocalsInit]
    public T AverageCost(TrainingSession<T> trainingSession)
    {
        T   totalCost          = T.AdditiveIdentity;
        var trainingDataPoints = trainingSession.TrainingDataPoints;
        var inferenceSession   = trainingSession.InferenceSession;

        foreach (var point in trainingDataPoints)
        {
            var inference = RunInferenceCore(inferenceSession, point.Inputs.Span);
            totalCost += trainingSession.CalculateCost(point.ExpectedOutputs.Span, inference);
        }

        return totalCost / T.CreateChecked(trainingDataPoints.Length);
    }

    private ReadOnlySpan<T> RunInferenceCore(InferenceSession<T> inferenceSession, ReadOnlySpan<T> inputs)
    {
        inputs.CopyTo(inferenceSession.Input);

        ref var layer     = ref _layers.Ref();
        ref var layersEnd = ref Unsafe.Add(ref layer, _layers.Length);
        while (Unsafe.IsAddressLessThan(ref layer, ref layersEnd))
        {
            layer.RunInference(inferenceSession.Input[..layer.Inputs], inferenceSession.Output[..layer.Size]);
            inferenceSession.Swap();

            layer = ref Unsafe.Add(ref layer, 1)!;
        }

        // After the last iteration, the output buffer gets swapped with the input,
        // so we copy it to the final output.
        return inferenceSession.Input[..Outputs];
    }

    public void Train(TrainingSession<T> trainingSession, T learnRate)
    {
        AverageCostFunc<T> costFunction = AverageCost;
        var                originalCost = costFunction(trainingSession);

        ref var layer     = ref _layers.Ref();
        ref var layersEnd = ref Unsafe.Add(ref layer, _layers.Length);
        while (Unsafe.IsAddressLessThan(ref layer, ref layersEnd))
        {
            layer.CalculateCostGradients(trainingSession, costFunction, originalCost);

            layer = ref Unsafe.Add(ref layer, 1)!;
        }

        layer = ref _layers.Ref()!;
        while (Unsafe.IsAddressLessThan(ref layer, ref layersEnd))
        {
            layer.ApplyCostGradients(trainingSession, learnRate);

            layer = ref Unsafe.Add(ref layer, 1)!;
        }
    }
}

public delegate T AverageCostFunc<T>(TrainingSession<T> trainingSession);
