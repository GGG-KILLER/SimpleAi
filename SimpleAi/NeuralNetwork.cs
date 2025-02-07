using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SimpleAi;

public sealed class NeuralNetwork<T>
    where T : unmanaged, INumber<T>
{
    private readonly Layer<T>[] _layers;

    public int Inputs { get; }

    public int Outputs { get; }

    public ReadOnlySpan<Layer<T>> Layers => _layers.AsSpan();

    public NeuralNetwork(int inputs, params ReadOnlySpan<int> layerSizes)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputs);
        if (layerSizes.Length < 1)
        {
            throw new ArgumentException("At least one layer is required.", nameof(layerSizes));
        }

        Inputs = inputs;
        Outputs = layerSizes[^1];
        _layers = new Layer<T>[layerSizes.Length];

        for (var idx = 0; idx < layerSizes.Length; idx++)
        {
            if (layerSizes[idx] <= 0)
            {
                throw new ArgumentException("Layer sizes must be greater than zero.", nameof(layerSizes));
            }

            _layers[idx] = new Layer<T>(idx == 0 ? inputs : layerSizes[idx - 1], layerSizes[idx]);
        }
    }

    private NeuralNetwork(Layer<T>[] layers)
    {
        _layers = layers;
        Inputs = layers[0].Inputs;
        Outputs = layers[^1].Size;
    }

    public static NeuralNetwork<T> UnsafeLoad(T[][] weights, T[][] biases)
    {
        var layers = new Layer<T>[weights.Length];
        for (var idx = 0; idx < weights.Length; idx++)
        {
            layers[idx] = Layer<T>.LoadUnsafe(weights[idx], biases[idx]);
        }
        return new NeuralNetwork<T>(layers);
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
}
