using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
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

        Debug.Assert(inputs.Length == inputCount, "Inputs are not the correct size.");
        Debug.Assert(outputs.Length >= neuronCount, "Outputs are not the correct size.");

        ref T weights = ref _weights.Ref();

        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported && inputs.Length > Vector<T>.Count * 2)
        {
            var idx = 0;

            Span<Vector<T>> neuronVecAccs = neuronCount < 16
                ? stackalloc Vector<T>[neuronCount]
                : new Vector<T>[neuronCount];
            neuronVecAccs.Fill(Vector<T>.Zero);

            for (; idx < inputs.Length - Vector<T>.Count; idx += Vector<T>.Count)
            {
                var leftVec = Vector.LoadUnsafe(ref inputs.UnsafeIndex(idx));
                for (var neuronIdx = 0; neuronIdx < Size; neuronIdx++)
                {
                    ref var vecAcc = ref neuronVecAccs.UnsafeIndex(neuronIdx);
                    var rightVec = Vector.LoadUnsafe(ref Unsafe.Add(ref weights, neuronIdx * Inputs + idx));

                    if (typeof(T) == typeof(double))
                    {
                        vecAcc = Vector.FusedMultiplyAdd(
                            leftVec.As<T, double>(),
                            rightVec.As<T, double>(),
                            vecAcc.As<T, double>()).As<double, T>();
                    }
                    else if (typeof(T) == typeof(float))
                    {
                        vecAcc = Vector.FusedMultiplyAdd(
                            leftVec.As<T, float>(),
                            rightVec.As<T, float>(),
                            vecAcc.As<T, float>()).As<float, T>();
                    }
                    else
                    {
                        vecAcc += leftVec * rightVec;
                    }
                }
            }

            var slowStart = idx;
            for (var neuronIdx = 0; neuronIdx < Size; neuronIdx++)
            {
                var acc = Vector.Sum(neuronVecAccs.UnsafeIndex(neuronIdx));

                for (idx = slowStart; idx < inputCount; idx++)
                {
                    var leftNum = inputs.UnsafeIndex(idx);
                    var rightNum = Unsafe.Add(ref weights, neuronIdx * inputCount + idx);
                    acc += leftNum * rightNum;
                }

                outputs.UnsafeIndex(neuronIdx) = acc;
            }

            MathEx.Binary<T, BUPipeline<T, AddOp<T>, ReLUOp<T>>>(outputs, _biases, outputs);
        }
        else // Software fallback
        {
            var biases = _biases;

            for (var neuronIdx = 0; neuronIdx < Size; neuronIdx++)
            {
                var acc = T.Zero;

                for (var idx = 0; idx < inputCount; idx++)
                {
                    var input = inputs.UnsafeIndex(idx);
                    var weight = Unsafe.Add(ref weights, neuronIdx * inputCount + idx);
                    acc += input * weight;
                }

                outputs.UnsafeIndex(neuronIdx) = MathEx.ReLU(acc + biases.UnsafeIndex(neuronIdx));
            }
        }
    }
}
