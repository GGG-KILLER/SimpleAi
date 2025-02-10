using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SimpleAi.Math;

namespace SimpleAi;

public interface ILayer<T>
{
    int Inputs { get; }
    int Size { get; }

    // Init
    void Randomize(T scale);

    // Inference
    void RunInference(ReadOnlySpan<T> inputs, Span<T> outputs);

    // Training
    T AverageCost(ReadOnlySpan<TrainingDataPoint<T>> trainingDataPoints);
}

public sealed class Layer<T, TActivation, TCost> : ILayer<T>
    where T : unmanaged, INumber<T>
    where TActivation : IActivationFunction<T>
    where TCost : ICostFunction<T>
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

    public static Layer<T, TActivation, TCost> LoadUnsafe(T[] weights, T[] biases)
    {
        var layer = new Layer<T, TActivation, TCost>(weights.Length / biases.Length, biases.Length);

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

        MathEx.Binary<T, AddOp<T>>(outputs, _biases, outputs);
        TActivation.Activate(outputs, outputs);
    }

    [SkipLocalsInit]
    public T AverageCost(ReadOnlySpan<TrainingDataPoint<T>> trainingDataPoints)
    {
        Span<T> results = Size <= (1024 / Marshal.SizeOf<T>())
            ? stackalloc T[Size]
            : GC.AllocateUninitializedArray<T>(Size);

        T totalCost = T.AdditiveIdentity;
        for (var idx = 0; idx < trainingDataPoints.Length; idx++)
        {
            var trainingDataPoint = trainingDataPoints.UnsafeIndex(idx);

            RunInference(trainingDataPoint.Inputs.Span, results);

            totalCost += TCost.Calculate(trainingDataPoint.ExpectedOutputs.Span, results);
        }
        return totalCost / T.CreateChecked(trainingDataPoints.Length);
    }
}
