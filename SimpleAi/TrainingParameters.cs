using JetBrains.Annotations;

namespace SimpleAi;

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
