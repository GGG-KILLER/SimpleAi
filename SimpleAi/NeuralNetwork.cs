using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;

namespace SimpleAi;

/// <summary>A class that represents a neural network with a variable amount of layers.</summary>
/// <typeparam name="T">The type of number that is used in the neural network.</typeparam>
[PublicAPI]
public sealed class NeuralNetwork<T> where T : IFloatingPoint<T>
{
    private readonly Layer<T>[] _layers;

    /// <summary>Initializes a new neural network instance.</summary>
    /// <param name="layers">The list of layers that will compose the neural network.</param>
    /// <exception cref="ArgumentException">
    ///     Thrown when no layer is provided or the layer layout is incorrect (inputs don't
    ///     match outputs of previous layer).
    /// </exception>
    [PublicAPI]
    public NeuralNetwork(params ReadOnlySpan<Layer<T>> layers)
    {
        if (layers.Length < 1)
            throw new ArgumentException(message: "At least one layer must be provided.", nameof(layers));

        _layers = layers.ToArray();
        Inputs  = layers[index: 0].Inputs;
        Outputs = layers[^1].Neurons;

        for (var idx = 1; idx < layers.Length; idx++)
        {
            if (layers[idx - 1].Neurons != layers[idx].Inputs)
                throw new ArgumentException($"Invalid layer layout. layers[{idx}].Inputs != layers[{idx - 1}].Neurons");
        }
    }

    /// <summary>The number of inputs this network accepts.</summary>
    public nint Inputs { get; }

    /// <summary>The number of outputs this network has.</summary>
    public nint Outputs { get; }

    /// <summary>The layers that compose this network.</summary>
    public ReadOnlySpan<Layer<T>> Layers => _layers.AsSpan();

    /// <summary>Executes inference on the provided inputs.</summary>
    /// <param name="inputs">The inputs to be processed by the network.</param>
    /// <exception cref="ArgumentException">
    ///     <list type="bullet">
    ///         <item>Thrown when <paramref name="inputs" /> does not have a Length of exactly <see cref="Inputs" />.</item>
    ///     </list>
    /// </exception>
    [SkipLocalsInit, PublicAPI]
    public Tensor<T> RunInference(in ReadOnlyTensorSpan<T> inputs)
    {
        if (!inputs.Lengths.SequenceEqual([Inputs]))
            throw new ArgumentException(
                message: "Input vector does not have the correct amount of elements.",
                nameof(inputs));

        // Run first layer
        Tensor<T> output = _layers[0].RunInference(inputs);
        // Process remaining layers until output
        for (var idx = 1; idx < _layers.Length; idx++)
        {
            output = _layers[idx].RunInference(output);
        }
        return output;
    }
}
