using System.Collections;
using JetBrains.Annotations;

namespace SimpleAi;

/// <summary>
/// The base interface for a training data storage.
/// </summary>
/// <typeparam name="T">The numeric type used by the <see cref="NeuralNetwork{T}"/></typeparam>
[PublicAPI]
public interface ITrainingData<T> : IEnumerable<TrainingDataPoint<T>>
{
    /// <summary>
    /// The total number of items contained within this provider.
    /// </summary>
    [PublicAPI]
    int Length { get; }

    /// <summary>
    /// Obtains a single training data point.
    /// </summary>
    /// <param name="index">The index to obtain data at.</param>
    [PublicAPI]
    TrainingDataPoint<T> this[Index index] { get; }

    /// <summary>
    /// Obtains a slice of the training data.
    /// </summary>
    /// <param name="range">The range of values to obtain data from.</param>
    [PublicAPI]
    ReadOnlyMemory<TrainingDataPoint<T>> this[Range range] { get; }

    /// <summary>
    /// Shuffles the training data randomly.
    /// </summary>
    [PublicAPI]
    void Shuffle();
}

/// <summary>
/// A training data storage stored entirely in memory without any special processing.
/// </summary>
/// <remarks>
/// The constructor creates a copy of the data to prevent mutations.
/// </remarks>
/// <param name="trainingData">The training data array to copy data from.</param>
/// <typeparam name="T">The numeric type used by the <see cref="NeuralNetwork{T}"/></typeparam>
[PublicAPI]
public sealed class InMemoryTrainingData<T>(IEnumerable<TrainingDataPoint<T>> trainingData) : ITrainingData<T>
{
    private readonly TrainingDataPoint<T>[] _trainingData = [..trainingData];

    /// <inheritdoc />
    public int Length => _trainingData.Length;

    /// <inheritdoc />
    public TrainingDataPoint<T> this[Index index] => _trainingData[index];

    /// <inheritdoc />
    public ReadOnlyMemory<TrainingDataPoint<T>> this[Range range] => _trainingData.AsMemory()[range];

    /// <inheritdoc />
    public void Shuffle() => Random.Shared.Shuffle(_trainingData);

    /// <summary>
    /// Returns an enumerator that enumerates through the contents of the training data.
    /// </summary>
    /// <returns>An enumerator.</returns>
    public IEnumerator<TrainingDataPoint<T>> GetEnumerator()
        => ((IEnumerable<TrainingDataPoint<T>>) _trainingData).GetEnumerator();

    /// <inheritdoc cref="GetEnumerator" />
    IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable) _trainingData).GetEnumerator();
}
