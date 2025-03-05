using JetBrains.Annotations;

namespace SimpleAi;

[PublicAPI]
public readonly record struct TrainingDataPoint<T>(T[] Inputs, T[] ExpectedOutputs);
