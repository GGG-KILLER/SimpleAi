using JetBrains.Annotations;

namespace SimpleAi;

[PublicAPI]
public readonly struct Weights<T>
{
    private readonly T[] _weights;

    public Weights(T[] weights, int inputs)
    {
        if (weights.Length % inputs != 0)
            throw new ArgumentException("Input weights are not a perfect multiple of inputs.", nameof(weights));

        _weights = weights;
        Inputs   = inputs;
    }

    [PublicAPI]
    public int Nodes => _weights.Length / Inputs;

    [PublicAPI]
    public int Inputs { get; }

    [PublicAPI]
    public int Length => _weights.Length;

    [PublicAPI]
    public ref T this[int index] => ref _weights.UnsafeIndex(index);

    [PublicAPI]
    public ref T this[int nodeIndex, int inputIndex] => ref _weights.UnsafeIndex(nodeIndex * Inputs + inputIndex);

    [PublicAPI]
    public Span<T> GetNodeWeights(int nodeIndex) => _weights.AsSpan(nodeIndex * Inputs, Inputs);

    [PublicAPI]
    public Span<T> AsSpan() => _weights.AsSpan();

    [PublicAPI]
    public static implicit operator ReadOnlyWeights<T>(Weights<T> weights)
        => new(weights._weights, weights.Inputs);
}

[PublicAPI]
public readonly struct ReadOnlyWeights<T>
{
    private readonly T[] _weights;
    private readonly int _inputs;

    public ReadOnlyWeights(T[] weights, int inputs)
    {
        if (weights.Length % inputs != 0)
            throw new ArgumentException("Input weights are not a perfect multiple of inputs.", nameof(weights));

        _weights = weights;
        _inputs  = inputs;
    }

    [PublicAPI]
    public int Nodes => _weights.Length / _inputs;

    [PublicAPI]
    public int Inputs => _inputs;

    [PublicAPI]
    public int Length => _weights.Length;

    [PublicAPI]
    public ref T this[int index] => ref _weights.UnsafeIndex(index);

    [PublicAPI]
    public T this[int nodeIndex, int inputIndex] => _weights.UnsafeIndex(nodeIndex * _inputs + inputIndex);

    [PublicAPI]
    public ReadOnlySpan<T> GetNodeWeights(int nodeIndex) => _weights.AsSpan(nodeIndex * Inputs, Inputs);

    [PublicAPI]
    public ReadOnlySpan<T> AsSpan() => _weights.AsSpan();
}
