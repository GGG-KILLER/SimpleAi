using System.Numerics;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;

namespace SimpleAi;

/// <summary>
/// A class that represents a neural network with a variable amount of layers.
/// </summary>
/// <typeparam name="T">The type of number that is used in the neural network.</typeparam>
[PublicAPI]
public sealed class NeuralNetwork<T> where T : unmanaged, INumber<T>
{
    private readonly Layer<T>[] _layers;

    /// <summary>
    /// Initializes a new neural network instance.
    /// </summary>
    /// <param name="layers">The list of layers that will compose the neural network.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when no layer is provided or the layer layout is incorrect (inputs don't match outputs of previous layer).
    /// </exception>
    [PublicAPI]
    public NeuralNetwork(params ReadOnlySpan<Layer<T>> layers)
    {
        if (layers.Length < 1) throw new ArgumentException("At least one layer must be provided.", nameof(layers));

        _layers = layers.ToArray();
        Inputs  = layers[0].Inputs;
        Outputs = layers[^1].Outputs;

        for (var idx = 1; idx < layers.Length; idx++)
        {
            if (layers[idx - 1].Outputs != layers[idx].Inputs)
                throw new ArgumentException($"Invalid layer layout. layers[{idx}].Inputs != layers[{idx - 1}].Outputs");
        }
    }

    /// <summary>
    /// The number of inputs this network accepts.
    /// </summary>
    public int Inputs { get; }

    /// <summary>
    /// The number of outputs this network has.
    /// </summary>
    public int Outputs { get; }

    /// <summary>
    /// The layers that compose this network.
    /// </summary>
    public ReadOnlySpan<Layer<T>> Layers => _layers.AsSpan();

    /// <summary>
    ///     <para>Randomizes the weights of the network using a normal distribution.</para>
    ///     <para>Will generate weights that are <paramref name="mean"/> ± <paramref name="stdDev"/>.</para>
    /// </summary>
    /// <param name="mean">The middle point of the weight distribution.</param>
    /// <param name="stdDev">The margin of randomness to allow around the <paramref name="mean"/>.</param>
    [PublicAPI]
    public void RandomizeWeights(T mean, T stdDev)
    {
        for (var idx = 0; idx < _layers.Length; idx++) _layers.UnsafeIndex(idx).RandomizeWeights(mean, stdDev);
    }

    /// <summary>
    /// Executes inference on the provided inputs.
    /// </summary>
    /// <param name="inferenceBuffer">The session associated with this inference operation.</param>
    /// <param name="inputs">The inputs to be processed by the network.</param>
    /// <param name="output">The network's outputs.</param>
    /// <exception cref="ArgumentException">
    /// <list type="bullet">
    ///     <item>Thrown when <paramref name="inputs"/> does not have a Length of exactly <see cref="Inputs"/>.</item>
    ///     <item>Thrown when <paramref name="output"/> does not have a Length of exactly <see cref="Outputs"/>.</item>
    /// </list>
    /// </exception>
    [SkipLocalsInit, PublicAPI]
    public void RunInference(InferenceBuffer<T> inferenceBuffer, ReadOnlySpan<T> inputs, Span<T> output)
    {
        if (inputs.Length != Inputs)
            throw new ArgumentException(
                message: "Input vector does not have the correct amount of elements.",
                nameof(inputs));
        if (output.Length != Outputs)
            throw new ArgumentException(message: "Output does not have the correct size.", nameof(output));

        ReadOnlySpan<T> inference = RunInferenceCore(inferenceBuffer, inputs);
        inference.CopyTo(output);
    }

    internal ReadOnlySpan<T> RunInferenceCore(InferenceBuffer<T> inferenceBuffer, ReadOnlySpan<T> inputs)
    {
        inputs.CopyTo(inferenceBuffer.Input);

        ref Layer<T> layer     = ref _layers.Ref();
        ref Layer<T> layersEnd = ref Unsafe.Add(ref layer, _layers.Length);
        while (Unsafe.IsAddressLessThan(ref layer, ref layersEnd))
        {
            layer.RunInference(inferenceBuffer.Input[..layer.Inputs], inferenceBuffer.Output[..layer.Outputs]);
            inferenceBuffer.Swap();

            layer = ref Unsafe.Add(ref layer, elementOffset: 1);
        }

        // After the last iteration, the output buffer gets swapped with the input, so we copy it to the final output.
        return inferenceBuffer.Input[..Outputs];
    }
}
