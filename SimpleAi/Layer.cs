using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SimpleAi.Math;

namespace SimpleAi;

public sealed class Layer<T>
    where T : INumber<T>
{
    private readonly T[] _weights;
    private readonly T[] _biases;

    public int Inputs { get; }
    public int Size { get; }

    public Layer(int inputCount, int size)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(size);

        _weights = new T[size * inputCount];
        _biases = new T[size];

        Inputs = inputCount;
        Size = size;
    }

    public static Layer<T> LoadUnsafe(T[] weights, T[] biases)
    {
        var layer = new Layer<T>(weights.Length / biases.Length, biases.Length);

        weights.CopyTo(layer._weights.AsSpan());
        biases.CopyTo(layer._biases.AsSpan());

        return layer;
    }

    public void Randomize(T scale)
    {
        // TODO: Use a normal distribution RNG
        var scaleConv = double.CreateSaturating(scale);

        for (var nodeIdx = 0; nodeIdx < _weights.Length; nodeIdx++)
        {
            _weights[nodeIdx] = T.CreateSaturating(Random.Shared.NextDouble() * scaleConv);
        }

        for (var nodeIdx = 0; nodeIdx < _biases.Length; nodeIdx++)
        {
            _biases[nodeIdx] = T.CreateSaturating(Random.Shared.NextDouble() * scaleConv);
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

        MathEx.Binary<T, BUPipeline<T, AddOp<T>, ReLUOp<T>>>(outputs, _biases, outputs);
    }
}
