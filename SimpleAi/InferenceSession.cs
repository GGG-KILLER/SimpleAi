namespace SimpleAi;

public sealed class InferenceSession<T>
{
    private T[] _inputBuffer, _outputBuffer;

    public InferenceSession(INeuralNetwork<T> neuralNetwork)
    {
        var bufferSize = 0;
        for (var idx = 0; idx < neuralNetwork.LayerCount; idx++)
        {
            ILayer<T> layer = neuralNetwork[idx];

            bufferSize = int.Max(bufferSize, int.Max(layer.Inputs, layer.Size));
        }
        _inputBuffer  = GC.AllocateUninitializedArray<T>(bufferSize);
        _outputBuffer = GC.AllocateUninitializedArray<T>(bufferSize);
    }

    internal Span<T> Input => _inputBuffer;

    internal Span<T> Output => _outputBuffer;

    internal void Swap() => (_outputBuffer, _inputBuffer) = (_inputBuffer, _outputBuffer);
}
