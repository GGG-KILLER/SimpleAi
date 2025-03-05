using System.Numerics;
using System.Numerics.Tensors;
using JetBrains.Annotations;
using SimpleAi.Internal;

namespace SimpleAi;

/// <summary>Interface for the class responsible for doing the training for a neural network.</summary>
/// <typeparam name="T">The number type used by the neural network.</typeparam>
[PublicAPI]
public interface INetworkTrainer<T> where T : IFloatingPoint<T>
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

    /// <summary>The current learning rate.</summary>
    [PublicAPI]
    T LearningRate { get; }

    /// <summary>Returns the average cost for all training data.</summary>
    /// <returns>The average cost for all training data.</returns>
    [PublicAPI]
    T CalculateAverageCost();

    /// <summary>Executes a single training iteration using the provided learning rate.</summary>
    [PublicAPI]
    void RunTrainingIteration();
}

/// <summary>Parameters used while training the network.</summary>
/// <typeparam name="T">The number type used by the <see cref="NeuralNetwork{T}" />.</typeparam>
[PublicAPI]
public readonly record struct TrainingParameters<T>()
{
    /// <summary>The initial learning rate to use with the network.</summary>
    [PublicAPI]
    public required T InitialLearnRate { get; init; }

    /// <summary>Percentage at which the learning rate will be reduced after each epoch (in the 0 to 100 range).</summary>
    /// <remarks>
    ///     <para>The learning rate will be <c>100% - (epoch * decay)%</c> at any given point in training.</para>
    ///     <para>
    ///         So, for example, if the training is on its 3rd epoch, and the decay is <c>4%</c> per epoch, the learning rate
    ///         will be <c>92%</c> of the <see cref="InitialLearnRate" />.
    ///     </para>
    /// </remarks>
    [PublicAPI]
    public required T LearnRateDecay { get; init; }

    /// <summary>
    ///     <para>The size of each mini-batch, if mini-batch training should be used.</para>
    ///     <para>The default is to not use mini-batch training.</para>
    /// </summary>
    [PublicAPI]
    public int? BatchSize { get; init; } = null;

    /// <summary>
    ///     <para>Whether to shuffle the training data after each epoch.</para>
    ///     <para>Default: true</para>
    /// </summary>
    [PublicAPI]
    public bool ShuffleDataAfterEpoch { get; init; } = true;

    /// <summary>
    ///     <para>Whether to run training in parallel using multiple threads.</para>
    ///     <para>Not recommended if your network is small or there isn't much training data in each mini-batch.</para>
    ///     <para>Default: true</para>
    /// </summary>
    [PublicAPI]
    public bool ParallelizeTraining { get; init; } = true;
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
    where T : IFloatingPoint<T> where TCost : ICostFunction<T>
{
    private readonly int                                  _batchCount, _derivativeArraySize;
    private readonly LayerCostGradients[]                 _costGradients;
    private readonly ObjectPool<LayerInferenceData[]>     _layerDataPool;
    private readonly NeuralNetwork<T>                     _network;
    private readonly MemoryIterator<TrainingDataPoint<T>> _trainingDataBatchIterator = new();
    private readonly TrainingParameters<T>                _trainingParameters;
    private          int                                  _iteration;

    /// <summary>Initializes a new neural network trainer.</summary>
    /// <param name="network">The neural network that will be trained.</param>
    /// <param name="trainingData">The training data to be used in training.</param>
    /// <param name="trainingParameters"></param>
    /// <exception cref="ArgumentException">
    ///     <list type="bullet">
    ///         <item>Thrown if the provided network is null.</item>
    ///         <item>Thrown if the provided training data is null, empty or invalid.</item>
    ///     </list>
    /// </exception>
    [PublicAPI]
    public NetworkTrainer(
        NeuralNetwork<T>      network,
        ITrainingData<T>      trainingData,
        TrainingParameters<T> trainingParameters)
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

        _network            = network;
        _trainingParameters = trainingParameters;
        TrainingData        = trainingData;
        BatchSize           = trainingParameters.BatchSize.GetValueOrDefault(TrainingData.Length);
        LearningRate        = trainingParameters.InitialLearnRate;
        _batchCount         = DivideRoundingUp(TrainingData.Length, BatchSize);

        _layerDataPool = new ObjectPool<LayerInferenceData[]>(
            CreateLayerDataArray,
            trainingParameters.ParallelizeTraining ? Environment.ProcessorCount : 1);

        ReadOnlySpan<Layer<T>> layers        = _network.Layers;
        var                    costGradients = new LayerCostGradients[layers.Length];
        for (var idx = 0; idx < costGradients.Length; idx++)
        {
            Layer<T> layer = layers[idx];
            costGradients[idx] = new LayerCostGradients(
                Tensor.Create<T>([layer.Neurons, layer.Inputs]),
                Tensor.Create<T>([layer.Neurons]));
        }
        _costGradients = costGradients;
        return;

        static int DivideRoundingUp(int dividend, int divisor) => (dividend + divisor - 1) / divisor;
    }

    /// <inheritdoc cref="NetworkTrainer(NeuralNetwork{T},ITrainingData{T},int,bool,bool)" />
    /// <param name="trainingDataPoints">The data points to be used in training.</param>
    [PublicAPI]
    public NetworkTrainer(
        NeuralNetwork<T>                  network,
        IEnumerable<TrainingDataPoint<T>> trainingDataPoints,
        TrainingParameters<T>             trainingParameters) : this(
        network,
        new InMemoryTrainingData<T>(trainingDataPoints ?? throw new ArgumentNullException(nameof(trainingDataPoints))),
        trainingParameters) { }

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
    [PublicAPI]
    public T LearningRate { get; private set; }

    /// <inheritdoc />
    public ReadOnlyMemory<TrainingDataPoint<T>> CurrentBatch
        => TrainingData[((_iteration % _batchCount) * BatchSize)..Math.Min(
                            ((_iteration % _batchCount) * BatchSize) + BatchSize,
                            TrainingData.Length)];

    /// <inheritdoc />
    [PublicAPI]
    public T CalculateAverageCost()
    {
        T totalCost = T.AdditiveIdentity;
        foreach (TrainingDataPoint<T> point in TrainingData)
        {
            var inference = _network.RunInference((Tensor<T>) point.Inputs);
            totalCost += TCost.Calculate(point.ExpectedOutputs, inference);
        }
        return totalCost / T.CreateChecked(TrainingData.Length);
    }

    /// <inheritdoc />
    [PublicAPI]
    public void RunTrainingIteration()
    {
        // Clear cost gradients so they don't get interference from other runs.
        foreach (LayerCostGradients costGradient in _costGradients)
        {
            costGradient.ForWeights.Clear();
            costGradient.ForBiases.Clear();
        }

        if (_trainingParameters.ParallelizeTraining)
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
            Layer<T>           layer         = layers[idx];
            LayerCostGradients costGradients = _costGradients[idx];
            layer.ApplyCostGradients(
                costGradients.ForWeights,
                costGradients.ForBiases,
                BatchSize != TrainingData.Length ? LearningRate / T.CreateSaturating(_batchCount) : LearningRate);
        }

        _iteration += 1;

        (int epoch, int batch) = int.DivRem(_iteration, BatchSize);
        if (batch == 0)
        {
            LearningRate = _trainingParameters.InitialLearnRate
                           * (T.One / (T.One + (_trainingParameters.LearnRateDecay * T.CreateSaturating(epoch))));
            if (_trainingParameters.ParallelizeTraining) TrainingData.Shuffle();
        }
    }

    private void CalculateGradients(LayerInferenceData[] allLayersData, TrainingDataPoint<T> trainingDataPoint)
    {
        ReadOnlySpan<Layer<T>> layers = _network.Layers;

        Tensor<T> input = trainingDataPoint.Inputs;
        for (var idx = 0; idx < layers.Length; idx++)
        {
            Layer<T>           layer     = layers[idx];
            LayerInferenceData layerData = allLayersData[idx];

            input.CopyTo(layerData.Inputs);
            input = layer.RunInference(input, out var unactivatedOutputs);
            unactivatedOutputs.CopyTo(layerData.UnactivatedOutputs);
        }

        // Rename just to make it easier to follow for the rest of the code.
        var output = input;

        // Update output layer gradients
        Tensor<T> trailingDerivatives;
        {
            Layer<T>           outputLayer     = layers[^1];
            LayerInferenceData outputLayerData = allLayersData[^1];

            trailingDerivatives = GradientDescent.CalculateOutputLayerTrailingDerivatives<T, TCost>(
                outputLayer,
                trainingDataPoint.ExpectedOutputs,
                outputLayerData.UnactivatedOutputs,
                output);

            LayerCostGradients outputLayerGradients = _costGradients[^1];
            lock (outputLayerGradients.Lock)
            {
                GradientDescent.UpdateLayerGradients(
                    outputLayer,
                    outputLayerData.Inputs,
                    trailingDerivatives,
                    outputLayerGradients.ForWeights,
                    outputLayerGradients.ForBiases);
            }
        }

        // Update hidden layer gradients
        for (int idx = layers.Length - 2; idx >= 0; idx--)
        {
            Layer<T>           layer              = layers[idx];
            Layer<T>           nextLayer          = layers[idx + 1];
            LayerInferenceData layerInferenceData = allLayersData[idx];

            trailingDerivatives = GradientDescent.CalculateHiddenLayerTrailingDerivatives(
                layer,
                nextLayer,
                layerInferenceData.Inputs,
                layerInferenceData.UnactivatedOutputs,
                trailingDerivatives);

            LayerCostGradients layerGradients = _costGradients[idx];
            lock (layerGradients.Lock)
            {
                GradientDescent.UpdateLayerGradients(
                    layer,
                    layerInferenceData.Inputs,
                    trailingDerivatives,
                    layerGradients.ForWeights,
                    layerGradients.ForBiases);
            }
        }
    }

    private LayerInferenceData[] CreateLayerDataArray()
    {
        ReadOnlySpan<Layer<T>> layers    = _network.Layers;
        LayerInferenceData[]   layerData = GC.AllocateUninitializedArray<LayerInferenceData>(layers.Length);
        for (var idx = 0; idx < layers.Length; idx++)
        {
            Layer<T> layer = layers[idx];
            layerData[idx] = new LayerInferenceData(
                Tensor.Create<T>([layer.Inputs]),
                Tensor.Create<T>([layer.Neurons]));
        }
        return layerData;
    }

    private readonly record struct LayerInferenceData(Tensor<T> Inputs, Tensor<T> UnactivatedOutputs);

    private readonly record struct LayerCostGradients(Tensor<T> ForWeights, Tensor<T> ForBiases)
    {
        public object Lock { get; } = new();
    }
}
