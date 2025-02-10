namespace SimpleAi;

public readonly record struct TrainingDataPoint<T>(ReadOnlyMemory<T> Inputs, ReadOnlyMemory<T> ExpectedOutputs);
