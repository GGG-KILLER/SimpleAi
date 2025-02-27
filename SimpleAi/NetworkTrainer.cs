using System.Numerics;
using JetBrains.Annotations;
using SimpleAi.Internal;
using SimpleAi.Math;

namespace SimpleAi;

/// <summary>Interface for the class responsible for doing the training for a neural network.</summary>
/// <typeparam name="T">The number type used by the neural network.</typeparam>
[PublicAPI]
public interface INetworkTrainer<T> where T : unmanaged, INumber<T>
{
    /// <summary>The training data being used for training.</summary>
    [PublicAPI]
    ITrainingData<T> TrainingData { get; }

    /// <summary>The size of the batches being used in training.</summary>
    [PublicAPI]
    int BatchSize { get; }

    /// <summary>The data points in the current training batch.</summary>
    [PublicAPI]
    ReadOnlyMemory<TrainingDataPoint<T>> CurrentBatch { get; }

    /// <summary>The current training epoch.</summary>
    [PublicAPI]
    double Epoch { get; }

    /// <summary>Returns the average cost for all training data.</summary>
    /// <returns>The average cost for all training data.</returns>
    [PublicAPI]
    T CalculateAverageCost();

    /// <summary>Executes a single training iteration using the provided learning rate.</summary>
    /// <param name="learnRate">The learning rate to use when training the network.</param>
    [PublicAPI]
    void RunTrainingIteration(T learnRate);
}

/// <summary>The class responsible for doing the training for a neural network.</summary>
/// <typeparam name="T">The number type used by the neural network.</typeparam>
/// <typeparam name="TCost">The cost function to be used in training.</typeparam>
/// <remarks>
///     This class creates some very large buffers that will probably end in the Large Object Heap. It is recommended
///     to do the training in a separate process that can be killed so the training buffers are not left in memory due to
///     being in the Large Object Heap.
/// </remarks>
[PublicAPI]
public sealed class NetworkTrainer<T, TCost> : INetworkTrainer<T>
    where T : unmanaged, INumber<T> where TCost : ICostFunction<T>
{
    private readonly int                                  _batchCount, _derivativeArraySize;
    private readonly LayerCostGradients[]                 _costGradients;
    private readonly ObjectPool<T[]>                      _derivativeArrayPool;
    private readonly ObjectPool<InferenceBuffer<T>>       _inferenceBufferPool;
    private readonly ObjectPool<LayerInferenceData[]>     _layerDataPool;
    private readonly NeuralNetwork<T>                     _network;
    private readonly bool                                 _shuffleDataPoints, _parallelizeTraining;
    private readonly MemoryIterator<TrainingDataPoint<T>> _trainingDataBatchIterator = new();
    private          int                                  _iteration;

    /// <summary>Initializes a new neural network trainer.</summary>
    /// <param name="network">The neural network that will be trained.</param>
    /// <param name="trainingData">The training data to be used in training.</param>
    /// <param name="batchSize">The size of the batch that should be used. Mini-batch mode will be disabled if 0 or negative.</param>
    /// <param name="shuffleDataPoints">Whether data points should be shuffled after every full training iteration.</param>
    /// <param name="parallelizeTraining">Whether training for all data points should be executed in parallel.</param>
    /// <exception cref="ArgumentException">
    ///     <list type="bullet">
    ///         <item>Thrown if the provided network is null.</item>
    ///         <item>Thrown if the provided training data is null, empty or invalid.</item>
    ///     </list>
    /// </exception>
    [PublicAPI]
    public NetworkTrainer(
        NeuralNetwork<T> network,
        ITrainingData<T> trainingData,
        int              batchSize           = -1,
        bool             shuffleDataPoints   = true,
        bool             parallelizeTraining = true)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(trainingData);
        if (trainingData.Length < 1)
            throw new ArgumentException(
                message: "At least one training data point must be provided.",
                nameof(trainingData));
        for (var idx = 0; idx < trainingData.Length; idx++)
        {
            TrainingDataPoint<T> trainingDataPoint = trainingData[idx];
            if (trainingDataPoint.Inputs.Length != network.Inputs)
                throw new ArgumentException($"Invalid training data point #{idx}. Input count mismatch.");
            if (trainingDataPoint.ExpectedOutputs.Length != network.Outputs)
                throw new ArgumentException($"Invalid training data point #{idx}. Output count mismatch.");
        }

        _network             = network;
        _shuffleDataPoints   = shuffleDataPoints;
        _parallelizeTraining = parallelizeTraining;
        TrainingData         = trainingData;
        BatchSize            = batchSize <= 0 ? TrainingData.Length : batchSize;
        _batchCount          = MathEx.DivideRoundingUp(TrainingData.Length, BatchSize);

        _derivativeArraySize = GradientDescent.GetTrailingDerivativesBufferSize(_network);
        _inferenceBufferPool = new ObjectPool<InferenceBuffer<T>>(
            CreateInferenceBuffer,
            parallelizeTraining ? Environment.ProcessorCount : 1);
        _layerDataPool = new ObjectPool<LayerInferenceData[]>(
            CreateLayerDataArray,
            parallelizeTraining ? Environment.ProcessorCount : 1);
        _derivativeArrayPool = new ObjectPool<T[]>(
            CreateDerivativeArray,
            parallelizeTraining ? Environment.ProcessorCount * 3 : 3);

        ReadOnlySpan<Layer<T>> layers        = _network.Layers;
        var                    costGradients = new LayerCostGradients[layers.Length];
        for (var idx = 0; idx < costGradients.Length; idx++)
        {
            Layer<T> layer = layers.UnsafeIndex(idx);
            costGradients.UnsafeIndex(idx) = new LayerCostGradients(
                new Weights<T>(GC.AllocateUninitializedArray<T>(layer.Inputs * layer.Outputs), layer.Inputs),
                GC.AllocateUninitializedArray<T>(layer.Outputs));
        }
        _costGradients = costGradients;
    }

    /// <inheritdoc cref="NetworkTrainer(NeuralNetwork{T},ITrainingData{T},int,bool,bool)" />
    /// <param name="trainingDataPoints">The data points to be used in training.</param>
    [PublicAPI]
    public NetworkTrainer(
        NeuralNetwork<T>                  network,
        IEnumerable<TrainingDataPoint<T>> trainingDataPoints,
        int                               batchSize           = -1,
        bool                              shuffleDataPoints   = true,
        bool                              parallelizeTraining = true) : this(
        network,
        new InMemoryTrainingData<T>(trainingDataPoints ?? throw new ArgumentNullException(nameof(trainingDataPoints))),
        batchSize,
        shuffleDataPoints,
        parallelizeTraining) { }

    /// <inheritdoc />
    [PublicAPI]
    public ITrainingData<T> TrainingData { get; }

    /// <inheritdoc />
    [PublicAPI]
    public int BatchSize { get; }

    /// <inheritdoc />
    [PublicAPI]
    public double Epoch => _iteration / (double) BatchSize;

    /// <inheritdoc />
    public ReadOnlyMemory<TrainingDataPoint<T>> CurrentBatch
        => TrainingData[((_iteration % _batchCount) * BatchSize)..System.Math.Min(
                            ((_iteration % _batchCount) * BatchSize) + BatchSize,
                            TrainingData.Length)];

    /// <inheritdoc />
    [PublicAPI]
    public T CalculateAverageCost()
    {
        T totalCost = T.AdditiveIdentity;

        InferenceBuffer<T> inferenceBuffer = _inferenceBufferPool.Rent();
        foreach (TrainingDataPoint<T> point in TrainingData)
        {
            ReadOnlySpan<T> inference = _network.RunInferenceCore(inferenceBuffer, point.Inputs.Span);
            totalCost += TCost.Calculate(point.ExpectedOutputs.Span, inference);
        }
        _inferenceBufferPool.Return(inferenceBuffer);

        return totalCost / T.CreateChecked(TrainingData.Length);
    }

    /// <inheritdoc />
    [PublicAPI]
    public void RunTrainingIteration(T learnRate)
    {
        // Clear cost gradients so they don't get interference from other runs.
        foreach (LayerCostGradients costGradient in _costGradients)
        {
            costGradient.ForWeights.AsSpan().Clear();
            costGradient.ForBiases.Clear();
        }

        if (_parallelizeTraining)
        {
            _trainingDataBatchIterator.Memory = CurrentBatch;
            Parallel.ForEach(
                _trainingDataBatchIterator,
                () => _layerDataPool.Rent(),
                (trainingDataPoint, _, tempLayerData) =>
                {
                    CalculateGradients(tempLayerData, trainingDataPoint);
                    return tempLayerData;
                },
                tempLayerData => _layerDataPool.Return(tempLayerData));
        }
        else
        {
            LayerInferenceData[] tempLayerData = _layerDataPool.Rent();
            foreach (TrainingDataPoint<T> trainingDataPoint in CurrentBatch.Span)
                CalculateGradients(tempLayerData, trainingDataPoint);
            _layerDataPool.Return(tempLayerData);
        }

        ReadOnlySpan<Layer<T>> layers = _network.Layers;
        for (var idx = 0; idx < layers.Length; idx++)
        {
            Layer<T>           layer         = layers.UnsafeIndex(idx);
            LayerCostGradients costGradients = _costGradients.UnsafeIndex(idx);
            layer.ApplyCostGradients(
                costGradients.ForWeights.AsSpan(),
                costGradients.ForBiases,
                BatchSize != TrainingData.Length ? learnRate / T.CreateSaturating(_batchCount) : learnRate);
        }

        _iteration += 1;
        if (_iteration % BatchSize == 0 && _shuffleDataPoints) TrainingData.Shuffle();
    }

    private void CalculateGradients(LayerInferenceData[] allLayersData, TrainingDataPoint<T> trainingDataPoint)
    {
        ReadOnlySpan<Layer<T>> layers = _network.Layers;

        InferenceBuffer<T> inferenceBuffer = _inferenceBufferPool.Rent();
        trainingDataPoint.Inputs.Span.CopyTo(inferenceBuffer.Input);
        for (var idx = 0; idx < layers.Length; idx++)
        {
            Layer<T>           layer     = layers.UnsafeIndex(idx);
            LayerInferenceData layerData = allLayersData.UnsafeIndex(idx);

            inferenceBuffer.Input[..layer.Inputs].CopyTo(layerData.Inputs);
            layer.RunInferenceCore(
                inferenceBuffer.Input[..layer.Inputs],
                inferenceBuffer.Output[..layer.Outputs],
                layerData.ActivationInputs);
            inferenceBuffer.Swap();
        }
        inferenceBuffer.Swap(); // Put the output back in its place

        // Update output layer gradients
        {
            Layer<T>           outputLayer           = layers.UnsafeIndex(^1);
            LayerInferenceData outputLayerData       = allLayersData.UnsafeIndex(^1);
            T[]                activationDerivatives = _derivativeArrayPool.Rent();
            T[]                costDerivatives       = _derivativeArrayPool.Rent();
            T[]                trailingDerivatives   = _derivativeArrayPool.Rent();

            GradientDescent.CalculateOutputLayerTrailingDerivatives<T, TCost>(
                outputLayer,
                trainingDataPoint.ExpectedOutputs.Span,
                outputLayerData.ActivationInputs,
                inferenceBuffer.Output[..outputLayer.Outputs],
                activationDerivatives,
                costDerivatives,
                trailingDerivatives);
            _derivativeArrayPool.Return(activationDerivatives);
            _derivativeArrayPool.Return(costDerivatives);
            _inferenceBufferPool.Return(inferenceBuffer);

            LayerCostGradients outputLayerGradients = _costGradients.UnsafeIndex(^1);
            lock (outputLayerGradients.Lock)
            {
                GradientDescent.UpdateLayerGradients(
                    outputLayer,
                    outputLayerData.Inputs,
                    trailingDerivatives,
                    outputLayerGradients.ForWeights,
                    outputLayerGradients.ForBiases);
            }
            _derivativeArrayPool.Return(trailingDerivatives);
        }

        // Update hidden layer gradients
        for (int idx = layers.Length - 2; idx >= 0; idx--)
        {
            Layer<T>           layer              = layers.UnsafeIndex(idx);
            Layer<T>           nextLayer          = layers.UnsafeIndex(idx + 1);
            LayerInferenceData layerInferenceData = allLayersData.UnsafeIndex(idx);

            T[] activationDerivatives = _derivativeArrayPool.Rent();
            T[] trailingDerivatives   = _derivativeArrayPool.Rent();
            GradientDescent.CalculateHiddenLayerTrailingDerivatives(
                layer,
                nextLayer,
                layerInferenceData.Inputs,
                layerInferenceData.ActivationInputs,
                activationDerivatives,
                trailingDerivatives);
            _derivativeArrayPool.Return(activationDerivatives);

            LayerCostGradients layerGradients = _costGradients.UnsafeIndex(idx);
            lock (layerGradients.Lock)
            {
                GradientDescent.UpdateLayerGradients(
                    layer,
                    layerInferenceData.Inputs,
                    trailingDerivatives,
                    layerGradients.ForWeights,
                    layerGradients.ForBiases);
            }
            _derivativeArrayPool.Return(trailingDerivatives);
        }
    }

    private InferenceBuffer<T> CreateInferenceBuffer() => new(_network);

    private LayerInferenceData[] CreateLayerDataArray()
    {
        ReadOnlySpan<Layer<T>> layers    = _network.Layers;
        LayerInferenceData[]   layerData = GC.AllocateUninitializedArray<LayerInferenceData>(layers.Length);
        for (var idx = 0; idx < layers.Length; idx++)
        {
            Layer<T> layer = layers[idx];
            layerData[idx] = new LayerInferenceData(
                GC.AllocateUninitializedArray<T>(layer.Inputs),
                GC.AllocateUninitializedArray<T>(layer.Outputs));
        }
        return layerData;
    }

    private T[] CreateDerivativeArray() => GC.AllocateUninitializedArray<T>(_derivativeArraySize);

    private readonly struct LayerInferenceData(T[] inputs, T[] activationInputs)
    {
        public Span<T> Inputs => inputs.AsSpan();

        public Span<T> ActivationInputs => activationInputs.AsSpan();
    }

    private readonly struct LayerCostGradients(Weights<T> forWeights, T[] forBiases)
    {
        public object Lock { get; } = new();

        public Weights<T> ForWeights => forWeights;

        public Span<T> ForBiases => forBiases.AsSpan();
    }
}
