using System.Numerics;
using JetBrains.Annotations;
using SimpleAi.Math;

namespace SimpleAi;

/// <summary>
/// Interface for the class responsible for doing the training for a neural network.
/// </summary>
/// <typeparam name="T">The number type used by the neural network.</typeparam>
[PublicAPI]
public interface INetworkTrainer<T> where T : unmanaged, INumber<T>
{
    /// <summary>
    /// The inference buffer used for training operations.
    /// </summary>
    /// <remarks>
    /// <para>This is shared for convenience for doing inferences during or after training.</para>
    /// <para>For inference only usage, create an inference buffer directly as it will require less memory.</para>
    /// </remarks>
    [PublicAPI]
    InferenceBuffer<T> InferenceBuffer { get; }

    /// <summary>
    /// All the data points being used for this training.
    /// </summary>
    [PublicAPI]
    IReadOnlyList<TrainingDataPoint<T>> TrainingDataPoints { get; }

    /// <summary>
    /// The size of the batches being used in training.
    /// </summary>
    [PublicAPI]
    int BatchSize { get; }

    /// <summary>
    /// The data points in the current training batch. 
    /// </summary>
    [PublicAPI]
    ReadOnlySpan<TrainingDataPoint<T>> CurrentBatch { get; }

    /// <summary>
    /// The current training epoch.
    /// </summary>
    [PublicAPI]
    double Epoch { get; }

    /// <summary>
    /// Returns the average cost for all training data.
    /// </summary>
    /// <returns>The average cost for all training data.</returns>
    [PublicAPI]
    T CalculateAverageCost();

    /// <summary>
    /// Executes a single training iteration using the provided learning rate.
    /// </summary>
    /// <param name="learnRate">The learning rate to use when training the network.</param>
    [PublicAPI]
    void RunTrainingIteration(T learnRate);
}

/// <summary>
/// The class responsible for doing the training for a neural network.
/// </summary>
/// <typeparam name="T">The number type used by the neural network.</typeparam>
/// <typeparam name="TCost">The cost function to be used in training.</typeparam>
/// <remarks>
/// This class creates some very large buffers that will probably end in the Large Object Heap.
/// It is recommended to do the training in a separate process that can be killed so the training buffers are not left
/// in memory due to being in the Large Object Heap.
/// </remarks>
[PublicAPI]
public sealed class NetworkTrainer<T, TCost> : INetworkTrainer<T>
    where T : unmanaged, INumber<T> where TCost : ICostFunction<T>
{
    private readonly NeuralNetwork<T>       _network;
    private readonly TrainingDataPoint<T>[] _trainingDataPoints;
    private readonly LayerData[]            _layerData;
    private readonly T[]                    _activationDerivatives, _costDerivatives, _trailingDerivatives;
    private readonly bool                   _shuffleDataPoints;
    private readonly int                    _batchCount;
    private          int                    _iteration;

    /// <summary>
    /// Initializes a new neural network trainer.
    /// </summary>
    /// <param name="network">The neural network that will be trained.</param>
    /// <param name="trainingDataPoints">The data points to be used in training.</param>
    /// <param name="batchSize">
    /// The size of the batch that should be used.
    /// Mini-batch mode will be disabled if 0 or negative.
    /// </param>
    /// <param name="shuffleDataPoints">Whether data points should be shuffled after every full training iteration.</param>
    /// <exception cref="ArgumentException">
    ///     <list type="bullet">
    ///         <item>Thrown if the provided network is null.</item>
    ///         <item>Thrown if the provided training data is null, empty or invalid.</item>
    ///     </list>
    /// </exception>
    [PublicAPI]
    public NetworkTrainer(
        NeuralNetwork<T>                  network,
        IEnumerable<TrainingDataPoint<T>> trainingDataPoints,
        int                               batchSize         = -1,
        bool                              shuffleDataPoints = true)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(trainingDataPoints);

        _network            = network;
        _shuffleDataPoints  = shuffleDataPoints;
        InferenceBuffer     = new InferenceBuffer<T>(network);
        _trainingDataPoints = [..trainingDataPoints];
        BatchSize           = batchSize <= 0 ? _trainingDataPoints.Length : batchSize;
        _batchCount         = MathEx.DivideRoundingUp(_trainingDataPoints.Length, BatchSize);
        if (_trainingDataPoints.Length < 1)
            throw new ArgumentException(
                "At least one training data point must be provided.",
                nameof(trainingDataPoints));

        for (var idx = 0; idx < _trainingDataPoints.Length; idx++)
        {
            var trainingDataPoint = _trainingDataPoints[idx];
            if (trainingDataPoint.Inputs.Length != network.Inputs)
                throw new ArgumentException($"Invalid training data point #{idx}. Input count mismatch.");
            if (trainingDataPoint.ExpectedOutputs.Length != network.Outputs)
                throw new ArgumentException($"Invalid training data point #{idx}. Output count mismatch.");
        }

        ReadOnlySpan<Layer<T>> layers = network.Layers;
        _layerData = GC.AllocateUninitializedArray<LayerData>(layers.Length);
        for (var idx = 0; idx < layers.Length; idx++)
        {
            var layer = layers[idx];
            _layerData[idx] = new LayerData(
                inputs: GC.AllocateUninitializedArray<T>(layer.Inputs),
                activationInputs: GC.AllocateUninitializedArray<T>(layer.Outputs),
                weightCostGradients: new Weights<T>(
                    GC.AllocateUninitializedArray<T>(layer.Inputs * layer.Outputs),
                    layer.Inputs),
                biasCostGradients: GC.AllocateUninitializedArray<T>(layer.Outputs));
        }

        int followingDerivativesSize = GradientDescent.GetTrailingDerivativesBufferSize(network);
        _activationDerivatives = GC.AllocateUninitializedArray<T>(followingDerivativesSize);
        _costDerivatives       = GC.AllocateUninitializedArray<T>(followingDerivativesSize);
        _trailingDerivatives   = GC.AllocateUninitializedArray<T>(followingDerivativesSize);
    }

    /// <inheritdoc />
    [PublicAPI]
    public InferenceBuffer<T> InferenceBuffer { get; }

    /// <inheritdoc />
    [PublicAPI]
    public IReadOnlyList<TrainingDataPoint<T>> TrainingDataPoints => _trainingDataPoints.AsReadOnly();

    /// <inheritdoc />
    [PublicAPI]
    public int BatchSize { get; }

    /// <inheritdoc />
    [PublicAPI]
    public double Epoch => _iteration / (double) BatchSize;

    /// <inheritdoc />
    public ReadOnlySpan<TrainingDataPoint<T>> CurrentBatch
        => _trainingDataPoints.AsSpan()[((_iteration % _batchCount) * BatchSize)..System.Math.Min(
                                            (_iteration % _batchCount) * BatchSize + BatchSize,
                                            _trainingDataPoints.Length)];

    /// <inheritdoc />
    [PublicAPI]
    public T CalculateAverageCost()
    {
        T                                  totalCost          = T.AdditiveIdentity;
        ReadOnlySpan<TrainingDataPoint<T>> trainingDataPoints = _trainingDataPoints;
        InferenceBuffer<T>                 inferenceBuffer    = InferenceBuffer;

        foreach (TrainingDataPoint<T> point in trainingDataPoints)
        {
            ReadOnlySpan<T> inference = _network.RunInferenceCore(inferenceBuffer, point.Inputs.Span);
            totalCost += TCost.Calculate(point.ExpectedOutputs.Span, inference);
        }

        return totalCost / T.CreateChecked(trainingDataPoints.Length);
    }

    private void CalculateGradients(TrainingDataPoint<T> trainingDataPoint)
    {
        ReadOnlySpan<Layer<T>> layers = _network.Layers;

        trainingDataPoint.Inputs.Span.CopyTo(InferenceBuffer.Input);
        for (var idx = 0; idx < layers.Length; idx++)
        {
            var layer     = layers.UnsafeIndex(idx);
            var layerData = _layerData.UnsafeIndex(idx);

            InferenceBuffer.Input[..layer.Inputs].CopyTo(layerData.Inputs);
            layer.RunInferenceCore(
                InferenceBuffer.Input[..layer.Inputs],
                InferenceBuffer.Output[..layer.Outputs],
                layerData.ActivationInputs);
            InferenceBuffer.Swap();
        }
        InferenceBuffer.Swap(); // Put the output back in its place

        // Update output layer gradients
        {
            var outputLayer     = layers.UnsafeIndex(^1);
            var outputLayerData = _layerData.UnsafeIndex(^1);
            GradientDescent.CalculateOutputLayerTrailingDerivatives<T, TCost>(
                outputLayer,
                trainingDataPoint.ExpectedOutputs.Span,
                outputLayerData.ActivationInputs,
                InferenceBuffer.Output[..outputLayer.Outputs],
                _activationDerivatives,
                _costDerivatives,
                _trailingDerivatives);
            GradientDescent.UpdateLayerGradients(
                outputLayer,
                outputLayerData.Inputs,
                _trailingDerivatives,
                outputLayerData.WeightCostGradients,
                outputLayerData.BiasCostGradients);
        }

        // Update hidden layer gradients
        for (int idx = layers.Length - 2; idx >= 0; idx--)
        {
            Layer<T>  layer     = layers.UnsafeIndex(idx);
            Layer<T>  nextLayer = layers.UnsafeIndex(idx + 1);
            LayerData layerData = _layerData.UnsafeIndex(idx);

            GradientDescent.CalculateHiddenLayerTrailingDerivatives(
                layer,
                nextLayer,
                layerData.Inputs,
                layerData.ActivationInputs,
                _activationDerivatives,
                _trailingDerivatives);

            GradientDescent.UpdateLayerGradients(
                layer,
                layerData.Inputs,
                _trailingDerivatives,
                layerData.WeightCostGradients,
                layerData.BiasCostGradients);
        }
    }

    /// <inheritdoc />
    [PublicAPI]
    public void RunTrainingIteration(T learnRate)
    {
        foreach (var trainingDataPoint in CurrentBatch)
        {
            CalculateGradients(trainingDataPoint);
        }

        ReadOnlySpan<Layer<T>> layers = _network.Layers;
        for (var idx = 0; idx < layers.Length; idx++)
        {
            var layer     = layers.UnsafeIndex(idx);
            var layerData = _layerData[idx];
            layer.ApplyCostGradients(
                layerData.WeightCostGradients.AsSpan(),
                layerData.BiasCostGradients,
                BatchSize != _trainingDataPoints.Length ? learnRate / T.CreateSaturating(_batchCount) : learnRate);
        }

        _iteration += 1;
        if (_iteration % BatchSize == 0 && _shuffleDataPoints) Random.Shared.Shuffle(_trainingDataPoints);
    }

    private readonly struct LayerData(
        T[]        inputs,
        T[]        activationInputs,
        Weights<T> weightCostGradients,
        T[]        biasCostGradients
    )
    {
        public Span<T> Inputs => inputs.AsSpan();

        public Span<T> ActivationInputs => activationInputs.AsSpan();

        public Weights<T> WeightCostGradients => weightCostGradients;

        public Span<T> BiasCostGradients => biasCostGradients.AsSpan();
    }
}
