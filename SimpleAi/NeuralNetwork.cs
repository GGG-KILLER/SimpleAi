using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SimpleAi;

public readonly struct NeuralNetwork<T>
    where T : INumber<T>, ISignedNumber<T>, IExponentialFunctions<T>
{
    private readonly Layer<T>[] _layers;

    public NeuralNetwork(int inputs, params ReadOnlySpan<int> layerSizes)
    {
        _layers = new Layer<T>[layerSizes.Length];

        for (var idx = 0; idx < layerSizes.Length; idx++)
        {
            _layers[idx] = new Layer<T>(idx == 0 ? inputs : layerSizes[idx - 1], layerSizes[idx]);
        }
    }
}

public readonly struct Layer<T>(int inputs, int nodes)
    where T : INumber<T>, ISignedNumber<T>, IExponentialFunctions<T>
{
    private readonly T[] _weights = new T[nodes * MathEx.DivideRoundingUp(inputs, Vector<T>.Count) * Vector<T>.Count];
    private readonly T[] _biases = new T[nodes];

    public int ExpectedInputs => inputs;
    public int ActualInputs { get; } = MathEx.DivideRoundingUp(inputs, Vector<T>.Count) * Vector<T>.Count;
    public int Nodes => nodes;

    public readonly void Randomize(T scale)
    {
        // TODO: Use a normal distribution RNG
        for (var nodeIdx = 0; nodeIdx < _weights.Length; nodeIdx++)
        {
            for (var inputIdx = 0; inputIdx < ActualInputs; inputIdx++)
            {
                if (inputIdx < ExpectedInputs)
                    _weights[nodeIdx * ActualInputs + inputIdx] = T.CreateSaturating(Random.Shared.NextDouble()) * scale;
                else
                    _weights[nodeIdx * ActualInputs + inputIdx] = T.Zero;
            }
        }

        for (var nodeIdx = 0; nodeIdx < _biases.Length; nodeIdx++)
        {
            _biases[nodeIdx] = T.CreateSaturating(Random.Shared.NextDouble()) * scale;
        }
    }

    public readonly void Calculate(ReadOnlySpan<T> inputs, Span<T> outputs)
    {
        Debug.Assert(inputs.Length == ActualInputs, "Inputs are not the correct size.");
        Debug.Assert(outputs.Length == Nodes, "Outputs are not the correct size.");

        ref T output = ref MemoryMarshal.GetReference(outputs);
        ref T outputEnd = ref Unsafe.Add(ref output, outputs.Length);
        ref T weights = ref MemoryMarshal.GetArrayDataReference(_weights);
        ref T weightsEnd = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(_weights), _weights.Length);
        ref T bias = ref MemoryMarshal.GetArrayDataReference(_biases);
        ref T biasEnd = ref Unsafe.Add(ref bias, _biases.Length);
        do
        {
            while (false) ;

            output = ref Unsafe.Add(ref output, 1);
            weights = ref Unsafe.Add(ref weights, inputs.Length);
            bias = ref Unsafe.Add(ref bias, 1);
        }
        while (Unsafe.IsAddressLessThan(ref weights, ref weightsEnd)
               && Unsafe.IsAddressLessThan(ref bias, ref biasEnd)
               && Unsafe.IsAddressLessThan(ref output, ref outputEnd));

        Debug.Assert(!Unsafe.IsAddressLessThan(ref weights, ref weightsEnd), "Did not use all weights.");
        Debug.Assert(!Unsafe.IsAddressLessThan(ref bias, ref biasEnd), "Did not use all biases.");
        Debug.Assert(!Unsafe.IsAddressLessThan(ref output, ref outputEnd), "Did not set all outputs.");
    }
}



internal static class MathEx
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int DivideRoundingUp(int number, int divisor) => (number + divisor - 1) / divisor;
}
