namespace SimpleAi;

public abstract class TrainingSession<T>
{
    private readonly Dictionary<ILayer<T>, LayerData> _layerData;
    private readonly Memory<TrainingDataPoint<T>>     _trainingData;

    protected TrainingSession(IEnumerable<TrainingDataPoint<T>> trainingDataPoints, INeuralNetwork<T> neuralNetwork)
    {
        _trainingData    = trainingDataPoints.ToArray();
        _layerData       = [];
        InferenceSession = new InferenceSession<T>(neuralNetwork);

        for (var idx = 0; idx < neuralNetwork.LayerCount; idx++)
        {
            ILayer<T> layer = neuralNetwork[idx];
            _layerData[layer] = new LayerData(
                GC.AllocateUninitializedArray<T>(layer.Size * layer.Inputs),
                GC.AllocateUninitializedArray<T>(layer.Size));
        }
    }

    public ReadOnlySpan<TrainingDataPoint<T>> TrainingDataPoints => _trainingData.Span;

    public InferenceSession<T> InferenceSession { get; }

    internal LayerData this[ILayer<T> layer] => _layerData[layer];

    public abstract T CalculateCost(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual);

    public void ShuffleTrainingData() => Random.Shared.Shuffle(_trainingData.Span);

    internal readonly record struct LayerData(Memory<T> WeightGradientCosts, Memory<T> BiasGradientCosts);
}

public sealed class TrainingSession<T, TCost>(
    IEnumerable<TrainingDataPoint<T>> trainingDataPoints,
    INeuralNetwork<T>                 neuralNetwork
) : TrainingSession<T>(trainingDataPoints, neuralNetwork) where TCost : ICostFunction<T>
{
    public override T CalculateCost(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual)
        => TCost.Calculate(expected, actual);
}
