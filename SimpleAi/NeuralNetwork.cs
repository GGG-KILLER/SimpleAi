using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SimpleAi;

public interface INeuralNetwork<T>
{
    int Inputs { get; }
    int Outputs { get; }

    ILayer<T> this[Index index] { get; }

    void Randomize(T scale);
    void RunInference(ReadOnlySpan<T> inputs, Span<T> output);

    T AverageCost(ReadOnlySpan<TrainingDataPoint<T>> trainingDataPoints);
}

public sealed class NeuralNetwork<T, TActivation, TCost> : INeuralNetwork<T>
    where T : unmanaged, INumber<T>
    where TActivation : IActivationFunction<T>
    where TCost : ICostFunction<T>
{
    private readonly Layer<T, TActivation, TCost>[] _layers;

    public int Inputs { get; }

    public int Outputs { get; }

    public ReadOnlySpan<Layer<T, TActivation, TCost>> Layers => _layers.AsSpan();

    public ILayer<T> this[Index index] => _layers[index];

    public NeuralNetwork(int inputs, params ReadOnlySpan<int> layerSizes)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputs);
        if (layerSizes.Length < 1)
        {
            throw new ArgumentException("At least one layer is required.", nameof(layerSizes));
        }

        Inputs = inputs;
        Outputs = layerSizes[^1];
        _layers = new Layer<T, TActivation, TCost>[layerSizes.Length];

        for (var idx = 0; idx < layerSizes.Length; idx++)
        {
            if (layerSizes[idx] <= 0)
            {
                throw new ArgumentException("Layer sizes must be greater than zero.", nameof(layerSizes));
            }

            _layers[idx] = new Layer<T, TActivation, TCost>(idx == 0 ? inputs : layerSizes[idx - 1], layerSizes[idx]);
        }
    }

    private NeuralNetwork(Layer<T, TActivation, TCost>[] layers)
    {
        _layers = layers;
        Inputs = layers[0].Inputs;
        Outputs = layers[^1].Size;
    }

    public static NeuralNetwork<T, TActivation, TCost> UnsafeLoad(T[][] weights, T[][] biases)
    {
        var layers = new Layer<T, TActivation, TCost>[weights.Length];
        for (var idx = 0; idx < weights.Length; idx++)
        {
            layers[idx] = Layer<T, TActivation, TCost>.LoadUnsafe(weights[idx], biases[idx]);
        }
        return new NeuralNetwork<T, TActivation, TCost>(layers);
    }

    public void Randomize(T scale)
    {
        for (int idx = 0; idx < _layers.Length; idx++)
        {
            _layers.UnsafeIndex(idx).Randomize(scale);
        }
    }

    [SkipLocalsInit]
    public void RunInference(ReadOnlySpan<T> inputs, Span<T> output)
    {
        if (inputs.Length != Inputs)
        {
            throw new ArgumentException("Input vector does not have the correct amount of elements.", nameof(inputs));
        }
        if (output.Length != Outputs)
        {
            throw new ArgumentException("Output does not have the correct size.", nameof(output));
        }

        var outputBuffersSize = _layers.Select(x => int.Max(x.Inputs, x.Size)).Max();

        Span<T> inBuffer = outputBuffersSize <= (1024 / Marshal.SizeOf<T>())
            ? stackalloc T[outputBuffersSize]
            : GC.AllocateUninitializedArray<T>(outputBuffersSize);
        Span<T> outBuffer = outputBuffersSize <= (1024 / Marshal.SizeOf<T>())
            ? stackalloc T[outputBuffersSize]
            : GC.AllocateUninitializedArray<T>(outputBuffersSize);

        inputs.CopyTo(inBuffer);
        for (var layerIdx = 0; layerIdx < _layers.Length; layerIdx++)
        {
            var layer = _layers.UnsafeIndex(layerIdx);

            layer.RunInference(inBuffer[..layer.Inputs], outBuffer[..layer.Size]);

            var tmp = outBuffer;
            outBuffer = inBuffer;
            inBuffer = tmp;
        }

        // After the last iteration, the output buffer gets swapped with the input,
        // so we copy it to the final output.
        inBuffer[..Outputs].CopyTo(output);
    }

    [SkipLocalsInit]
    public T AverageCost(ReadOnlySpan<TrainingDataPoint<T>> trainingDataPoints) =>
        _layers[^1].AverageCost(trainingDataPoints);
}
