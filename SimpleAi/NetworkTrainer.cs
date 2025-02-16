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
    /// Returns the average cost for the current training data batch.
    /// </summary>
    /// <returns>The average cost for the current training data batch.</returns>
    [PublicAPI]
    T CalculateAverageCost();

    /// <summary>
    /// Returns the average cost for all the training data.
    /// </summary>
    /// <returns>The average cost for the network for all the training data.</returns>
    [PublicAPI]
    T CalculateTotalAverageCost();

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
    private readonly bool                   _shuffleDataPoints;
    private readonly LayerData[]            _layerData;
    private readonly int                    _batchCount;
    private          int                    _iteration;
    private readonly TrainingDataPoint<T>[] _trainingDataPoints;

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
                GC.AllocateUninitializedArray<T>(layer.Inputs * layer.Outputs),
                GC.AllocateUninitializedArray<T>(layer.Outputs));
        }
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
    public ReadOnlySpan<TrainingDataPoint<T>> CurrentBatch
        => _trainingDataPoints.AsSpan()[(_iteration * BatchSize)..System.Math.Min(
                                            _iteration * BatchSize + BatchSize,
                                            _trainingDataPoints.Length)];

    /// <inheritdoc />
    [PublicAPI]
    public T CalculateAverageCost()
    {
        T                                  totalCost          = T.AdditiveIdentity;
        ReadOnlySpan<TrainingDataPoint<T>> trainingDataPoints = CurrentBatch;
        InferenceBuffer<T>                 inferenceBuffer    = InferenceBuffer;

        foreach (TrainingDataPoint<T> point in trainingDataPoints)
        {
            ReadOnlySpan<T> inference = _network.RunInferenceCore(inferenceBuffer, point.Inputs.Span);
            totalCost += TCost.Calculate(point.ExpectedOutputs.Span, inference);
        }

        return totalCost / T.CreateChecked(trainingDataPoints.Length);
    }

    /// <inheritdoc />
    [PublicAPI]
    public T CalculateTotalAverageCost()
    {
        T                                  totalCost          = T.AdditiveIdentity;
        ReadOnlySpan<TrainingDataPoint<T>> trainingDataPoints = _trainingDataPoints.AsSpan();
        InferenceBuffer<T>                 inferenceBuffer    = InferenceBuffer;

        foreach (TrainingDataPoint<T> point in trainingDataPoints)
        {
            ReadOnlySpan<T> inference = _network.RunInferenceCore(inferenceBuffer, point.Inputs.Span);
            totalCost += TCost.Calculate(point.ExpectedOutputs.Span, inference);
        }

        return totalCost / T.CreateChecked(trainingDataPoints.Length);
    }

    /// <inheritdoc />
    [PublicAPI]
    public void RunTrainingIteration(T learnRate)
    {
        ReadOnlySpan<Layer<T>> layers       = _network.Layers;
        T                      originalCost = CalculateAverageCost();

        Func<T> costFunction = CalculateAverageCost;
        for (var idx = 0; idx < layers.Length; idx++)
        {
            var layer     = layers.UnsafeIndex(idx);
            var layerData = _layerData[idx];
            layer.CalculateCostGradients(
                costFunction,
                originalCost,
                layerData.WeightGradientCosts.Span,
                layerData.BiasGradientCosts.Span);
        }

        for (var idx = 0; idx < layers.Length; idx++)
        {
            var layer     = layers.UnsafeIndex(idx);
            var layerData = _layerData[idx];
            layer.ApplyCostGradients(layerData.WeightGradientCosts.Span, layerData.BiasGradientCosts.Span, learnRate);
        }

        _iteration = (_iteration + 1) % _batchCount;
        if (_iteration == 0 && _shuffleDataPoints) Random.Shared.Shuffle(_trainingDataPoints);
    }

    private readonly record struct LayerData(Memory<T> WeightGradientCosts, Memory<T> BiasGradientCosts);
}
