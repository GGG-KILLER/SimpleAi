using JetBrains.Annotations;

namespace SimpleAi;

[PublicAPI]
public readonly record struct TrainingDataPoint<T>(ReadOnlyMemory<T> Inputs, ReadOnlyMemory<T> ExpectedOutputs);
