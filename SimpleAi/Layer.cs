using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi;

/// <summary>The base class for layers.</summary>
/// <typeparam name="T">The numeric type used by this layer.</typeparam>
/// <remarks>
///     <para>Activation must be done on the layer itself.</para>
/// </remarks>
[PublicAPI]
public abstract class Layer<T> where T : IFloatingPoint<T>
{
    protected readonly Tensor<T> MutBiases;
    protected readonly Tensor<T> MutWeights;

    /// <summary>Initializes a new instance of the layer.</summary>
    /// <param name="inputs">The number of inputs this layer will accept.</param>
    /// <param name="neurons">The number of neurons (outputs) this layer will have.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    ///     <list type="bullet">
    ///         <item>Thrown when <paramref name="inputs" /> is negative or zero.</item>
    ///         <item>Thrown when <paramref name="neurons" /> is negative or zero.</item>
    ///     </list>
    /// </exception>
    protected Layer(nint inputs, nint neurons)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputs);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(neurons);

        MutWeights = Tensor.CreateAndFillGaussianNormalDistribution<T>([neurons, inputs]);
        MutBiases  = Tensor.Create<T>([neurons]);

        Inputs  = inputs;
        Neurons = neurons;
    }

    /// <summary>
    /// Creates a new Layer from the provided <paramref name="weights"/> and <paramref name="biases"/>.
    /// </summary>
    /// <param name="weights">The weights to be used for this layer.</param>
    /// <param name="biases">The biases to be used for this layer.</param>
    /// <exception cref="ArgumentNullException">
    /// Thrown when any of the arguments are null.
    /// </exception>
    /// <exception cref="ArgumentException">
    ///     <list type="bullet">
    ///         <item>Thrown when <paramref name="weights"/> is not a 2D tensor.</item>
    ///         <item>Thrown when <paramref name="biases"/> is not a 1D tensor.</item>
    ///         <item>Thrown when <paramref name="weights"/>' 1st dimension is not the same size as <paramref name="biases"/>'.</item>
    ///     </list>
    /// </exception>
    protected Layer(Tensor<T> weights, Tensor<T> biases)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(biases);

        if (weights.Rank != 2) throw new ArgumentException("Weights must be a 2D tensor.", nameof(weights));
        if (biases.Rank != 1) throw new ArgumentException("Biases must be a 1D tensor.", nameof(biases));
        if (weights.FlattenedLength < 1)
            throw new ArgumentException("Layer must have at least one weight.", nameof(weights));
        if (biases.FlattenedLength < 1)
            throw new ArgumentException("Layer must have at least one bias.", nameof(biases));
        if (weights.Lengths[0] != biases.Lengths[0])
            throw new ArgumentException(
                "Biases must have the same size as the weights' 1st dimension.",
                nameof(weights));

        MutWeights = weights;
        MutBiases  = biases;

        Inputs  = weights.Lengths[1];
        Neurons = weights.Lengths[0];
    }

    /// <summary>Number of inputs this layer accepts.</summary>
    [PublicAPI]
    public nint Inputs { get; }

    /// <summary>Number of neurons this layer has.</summary>
    [PublicAPI]
    public nint Neurons { get; }

    /// <summary>The weights used by this layer.</summary>
    [PublicAPI]
    public ReadOnlyTensorSpan<T> Weights => MutWeights;

    /// <summary>The biases used by this layer.</summary>
    [PublicAPI]
    public ReadOnlyTensorSpan<T> Biases => MutBiases;

    /// <summary>
    /// The activation type being used by this layer.
    /// </summary>
    [PublicAPI]
    public abstract Type ActivationType { get; }

    /// <summary>Runs inference for this layer.</summary>
    /// <param name="inputs">The inputs to be processed by this layer.</param>
    /// <exception cref="ArgumentException">
    ///     <list type="bullet">
    ///         <item>Thrown when <paramref name="inputs" /> does not have a length of <b>at least</b> <see cref="Inputs" />.</item>
    ///     </list>
    /// </exception>
    [PublicAPI]
    public virtual Tensor<T> RunInference(in ReadOnlyTensorSpan<T> inputs) => RunInference(inputs, out _);

    /// <summary>Runs the inference process storing the unactivated outputs.</summary>
    /// <param name="inputs">The inputs to run inference on.</param>
    /// <param name="unactivatedOutputs">The return values but without the activation function executed on them.</param>
    [PublicAPI]
    public abstract Tensor<T> RunInference(in ReadOnlyTensorSpan<T> inputs, out Tensor<T> unactivatedOutputs);

    /// <summary>Calculates the partial derivatives for the activation.</summary>
    [PublicAPI]
    protected internal abstract Tensor<T> CalculateActivationDerivatives(in ReadOnlyTensorSpan<T> weightedInputs);

    /// <summary>Applies the supplied loss gradients to the weights and biases.</summary>
    /// <param name="weightGradientLosses">The loss gradients to apply to the weights.</param>
    /// <param name="biasGradientLosses">The bias loss gradients to apply to the biases.</param>
    /// <param name="learnRate">The learn rate to apply the loss gradients with.</param>
    [PublicAPI]
    protected internal virtual void ApplyLossGradients(
        in ReadOnlyTensorSpan<T> weightGradientLosses,
        in ReadOnlyTensorSpan<T> biasGradientLosses,
        T                        learnRate)
    {
        // weights -= weightGradientLosses * learnRate
        Tensor.Subtract<T>(MutWeights, Tensor.Multiply(weightGradientLosses, learnRate), MutWeights);

        // biases -= biasGradientLosses * learnRate
        Tensor.Subtract<T>(MutBiases, Tensor.Multiply(biasGradientLosses, learnRate), MutBiases);
    }
}

/// <summary>The actual implementation of the layer.</summary>
/// <typeparam name="T">The numeric type used by this layer.</typeparam>
/// <typeparam name="TActivation">The activation function type used by this layer.</typeparam>
[PublicAPI]
public sealed class Layer<T, TActivation> : Layer<T>
    where TActivation : IActivationFunction<T> where T : IFloatingPoint<T>
{
    /// <inheritdoc />
    public Layer(int inputs, int neurons) : base(inputs, neurons) { }

    /// <inheritdoc />
    public Layer(Tensor<T> weights, Tensor<T> biases) : base(weights, biases) { }

    /// <inheritdoc />
    public override Type ActivationType => typeof(TActivation);

    /// <inheritdoc />
    public override Tensor<T> RunInference(in ReadOnlyTensorSpan<T> inputs, out Tensor<T> unactivatedOutputs)
    {
        if (!inputs.Lengths.SequenceEqual([Inputs]))
            throw new ArgumentException($"Inputs are not the correct size.", nameof(inputs));

        unactivatedOutputs = Tensor.Create<T>([Neurons]);
        for (nint neuron = 0; neuron < Neurons; neuron++)
        {
            var neuronWeights = MutWeights[new NRange(neuron, neuron + 1), NRange.All];
            unactivatedOutputs[neuron] = Tensor.Sum<T>(Tensor.Multiply(neuronWeights, inputs)) + MutBiases[neuron];
        }
        return TActivation.Activate(unactivatedOutputs);
    }

    /// <inheritdoc />
    protected internal override Tensor<T> CalculateActivationDerivatives(in ReadOnlyTensorSpan<T> weightedInputs)
        => TActivation.Derivative(weightedInputs);
}
