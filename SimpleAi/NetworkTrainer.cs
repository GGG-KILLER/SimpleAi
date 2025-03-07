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

    /// <summary>Returns the average loss for all training data.</summary>
    /// <returns>The average loss for all training data.</returns>
    [PublicAPI]
    T CalculateAverageLoss();

    /// <summary>Executes a single training iteration using the provided learning rate.</summary>
    [PublicAPI]
    void RunTrainingIteration();
}

/// <summary>The class responsible for doing the training for a neural network.</summary>
/// <typeparam name="T">The number type used by the neural network.</typeparam>
/// <typeparam name="TLoss">The loss function to be used in training.</typeparam>
/// <remarks>
///     This class creates some very large buffers that will probably end in the Large Object Heap. It is recommended
///     to do the training in a separate process that can be killed so the training buffers are not left in memory due to
///     being in the Large Object Heap.
/// </remarks>
[PublicAPI]
public sealed class NetworkTrainer<T, TLoss> : INetworkTrainer<T>
    where T : IFloatingPoint<T> where TLoss : ILossFunction<T>
{
    private readonly int                                  _batchCount, _derivativeArraySize;
    private readonly LayerLossGradients[]                 _lossGradients;
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

        _lossGradients = CreateLossGradients();
        return;

        static int DivideRoundingUp(int dividend, int divisor) => (dividend + divisor - 1) / divisor;
    }

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
    public T CalculateAverageLoss()
    {
        if (_trainingParameters.ParallelizeTraining)
        {
            return TrainingData.AsParallel().Select(
                                   point => TLoss.Calculate(
                                       point.ExpectedOutputs,
                                       _network.RunInference((Tensor<T>) point.Inputs)))
                               .Aggregate(T.AdditiveIdentity, static (acc, next) => acc + next)
                   / T.CreateChecked(TrainingData.Length);
        }

        return TrainingData.Select(
                               point => TLoss.Calculate(
                                   point.ExpectedOutputs,
                                   _network.RunInference((Tensor<T>) point.Inputs)))
                           .Aggregate(T.AdditiveIdentity, static (acc, next) => acc + next)
               / T.CreateChecked(TrainingData.Length);
    }

    /// <inheritdoc />
    [PublicAPI]
    public void RunTrainingIteration()
    {
        // Clear loss gradients so they don't get interference from other runs.
        for (var idx = 0; idx < _lossGradients.Length; idx++)
        {
            Layer<T>           layer        = _network.Layers[idx];
            LayerLossGradients lossGradient = _lossGradients[idx];

            lossGradient.ForWeights ??= Tensor.Create<T>(layer.Weights.Lengths);
            lossGradient.ForWeights.Clear();
            lossGradient.ForBiases ??= Tensor.Create<T>(layer.Biases.Lengths);
            lossGradient.ForBiases.Clear();
        }

        LayerLossGradients[] lossGradients;
        _trainingDataBatchIterator.Memory = CurrentBatch;
        if (_trainingParameters.ParallelizeTraining)
        {
            lossGradients = _trainingDataBatchIterator.AsParallel().Select(CalculateGradients)
                                                      .Aggregate(_lossGradients, CombineLossGradients);
        }
        else
        {
            lossGradients = _trainingDataBatchIterator.Select(CalculateGradients)
                                                      .Aggregate(_lossGradients, CombineLossGradients);
        }

        ReadOnlySpan<Layer<T>> layers = _network.Layers;
        for (var idx = 0; idx < layers.Length; idx++)
        {
            Layer<T>           layer        = layers[idx];
            LayerLossGradients lossGradient = lossGradients[idx];
            layer.ApplyLossGradients(
                lossGradient.ForWeights!,
                lossGradient.ForBiases!,
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
        return;

        static LayerLossGradients[] CombineLossGradients(LayerLossGradients[] acc, LayerLossGradients[] next)
        {
            for (var idx = 0; idx < acc.Length; idx++)
            {
                var lossGradient      = acc[idx];
                var batchLossGradient = next[idx];
                Tensor.Add<T>(lossGradient.ForWeights!, batchLossGradient.ForWeights!, lossGradient.ForWeights!);
                Tensor.Add<T>(lossGradient.ForBiases!, batchLossGradient.ForBiases!, lossGradient.ForBiases!);
            }
            return acc;
        }
    }

    private LayerLossGradients[] CalculateGradients(TrainingDataPoint<T> trainingDataPoint)
    {
        LayerInferenceData[] allLayersData = _layerDataPool.Rent();
        try
        {
            var                    lossGradients = CreateLossGradients();
            ReadOnlySpan<Layer<T>> layers        = _network.Layers;

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

                trailingDerivatives = OptimizationHelper.CalculateOutputLayerTrailingDerivatives<T, TLoss>(
                    outputLayer,
                    trainingDataPoint.ExpectedOutputs,
                    outputLayerData.UnactivatedOutputs,
                    output);

                LayerLossGradients outputLayerGradients = lossGradients[^1];
                OptimizationHelper.CalculateLayerGradients(
                    outputLayer,
                    outputLayerData.Inputs,
                    trailingDerivatives,
                    out outputLayerGradients.ForWeights,
                    out outputLayerGradients.ForBiases);
            }

            // Update hidden layer gradients
            for (int idx = layers.Length - 2; idx >= 0; idx--)
            {
                Layer<T>           layer              = layers[idx];
                Layer<T>           nextLayer          = layers[idx + 1];
                LayerInferenceData layerInferenceData = allLayersData[idx];

                trailingDerivatives = OptimizationHelper.CalculateHiddenLayerTrailingDerivatives(
                    layer,
                    nextLayer,
                    layerInferenceData.Inputs,
                    layerInferenceData.UnactivatedOutputs,
                    trailingDerivatives);

                LayerLossGradients layerGradients = lossGradients[idx];
                OptimizationHelper.CalculateLayerGradients(
                    layer,
                    layerInferenceData.Inputs,
                    trailingDerivatives,
                    out layerGradients.ForWeights,
                    out layerGradients.ForBiases);
            }

            return lossGradients;
        }
        finally
        {
            _layerDataPool.Return(allLayersData);
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

    private LayerLossGradients[] CreateLossGradients()
    {
        var lossGradients = new LayerLossGradients[_network.Layers.Length];
        for (var idx = 0; idx < lossGradients.Length; idx++)
        {
            lossGradients[idx] = new LayerLossGradients();
        }
        return lossGradients;
    }

    private readonly record struct LayerInferenceData(Tensor<T> Inputs, Tensor<T> UnactivatedOutputs);

    private sealed class LayerLossGradients
    {
        public Tensor<T>? ForWeights, ForBiases;
    }
}
