using System.Numerics;
using JetBrains.Annotations;

namespace SimpleAi;

/// <summary>
///     <para>
///         A class holding the buffers used for inference operations.
///     </para>
///     <para>
///         An inference operation requires large buffers used in the inference, so this class exists to make them
///         reusable in the inference process to not let them up in the LOH without usage.
///     </para>
///     <para>
///         This class is <b>NOT THREAD-SAFE</b>. Different inference buffers should be used per <b>network and thread</b>.
///     </para>
/// </summary>
/// <remarks>
///     An inference session can only be used for the network it was created for and only by one thread at a time.
/// </remarks>
/// <typeparam name="T">The numeric type used in the neural network.</typeparam>
[PublicAPI]
public sealed class InferenceBuffer<T> where T : unmanaged, INumber<T>
{
    private T[] _inputBuffer, _outputBuffer;

    /// <summary>
    /// Initializes a new inference buffer.
    /// </summary>
    /// <param name="neuralNetwork">The neural network this inference session will be created for.</param>
    [PublicAPI]
    public InferenceBuffer(NeuralNetwork<T> neuralNetwork)
    {
        var bufferSize = 0;
        foreach (Layer<T> layer in neuralNetwork.Layers)
        {
            bufferSize = int.Max(bufferSize, int.Max(layer.Inputs, layer.Outputs));
        }
        _inputBuffer  = GC.AllocateUninitializedArray<T>(bufferSize);
        _outputBuffer = GC.AllocateUninitializedArray<T>(bufferSize);
    }

    internal Span<T> Input => _inputBuffer;

    internal Span<T> Output => _outputBuffer;

    internal void Swap() => (_outputBuffer, _inputBuffer) = (_inputBuffer, _outputBuffer);
}
