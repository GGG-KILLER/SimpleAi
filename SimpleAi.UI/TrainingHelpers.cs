using System;
using SimpleAi.UI.Maths;

namespace SimpleAi.UI;

internal static class TrainingHelpers
{
    private static ReadOnlyMemory<NumberTypeT> SafeOutput { get; } = (NumberTypeT[]) [1, 0];

    private static ReadOnlyMemory<NumberTypeT> UnsafeOutput { get; } = (NumberTypeT[]) [0, 1];

    public static TrainingDataPoint<NumberTypeT>[] GenerateTrainingData(
        Vector2DRange totalArea,
        Vector2DRange safeArea,
        int           unsafePoints,
        int           safePoints)
    {
        var trainingData       = new TrainingDataPoint<NumberTypeT>[safePoints + unsafePoints];
        var trainingDataInputs = new NumberTypeT[(safePoints + unsafePoints) * 2];
        var generatedSafe      = 0;
        var generatedUnsafe    = 0;

        for (var idx = 0; idx < trainingData.Length; idx++)
        {
            bool isSafe;
            if (generatedSafe < safePoints && generatedUnsafe < unsafePoints)
                isSafe = Random.Shared.NextDouble() > 0.5;
            else
                isSafe = generatedSafe < safePoints;

            if (isSafe)
            {
                trainingDataInputs[(idx * 2) + 0] = (NumberTypeT) (safeArea.Start.X
                                                                   + (Random.Shared.NextDouble()
                                                                      * (safeArea.End.X - safeArea.Start.X)));
                trainingDataInputs[(idx * 2) + 1] = (NumberTypeT) (safeArea.Start.Y
                                                                   + (Random.Shared.NextDouble()
                                                                      * (safeArea.End.Y - safeArea.Start.Y)));
                generatedSafe++;
            }
            else
            {
            regen:
                trainingDataInputs[(idx * 2) + 0] = (NumberTypeT) (totalArea.Start.X
                                                                   + (Random.Shared.NextDouble()
                                                                      * (totalArea.End.X - totalArea.Start.X)));
                trainingDataInputs[(idx * 2) + 1] = (NumberTypeT) (totalArea.Start.Y
                                                                   + (Random.Shared.NextDouble()
                                                                      * (totalArea.End.Y - totalArea.Start.Y)));

                if (safeArea.Start.X < trainingDataInputs[(idx * 2) + 0]
                    && trainingDataInputs[(idx * 2) + 0] < safeArea.End.X
                    && safeArea.Start.Y < trainingDataInputs[(idx * 2) + 1]
                    && trainingDataInputs[(idx * 2) + 1] < safeArea.End.Y)
                    goto regen;

                generatedUnsafe++;
            }

            trainingData[idx] = new TrainingDataPoint<NumberTypeT>(
                trainingDataInputs.AsMemory(idx * 2, length: 2),
                isSafe ? SafeOutput : UnsafeOutput);
        }

        return trainingData;
    }

    public static double CalculateAccuracy(
        this NeuralNetwork<NumberTypeT>  neuralNetwork,
        InferenceBuffer<NumberTypeT>     inferenceBuffer,
        TrainingDataPoint<NumberTypeT>[] checkDataPoints)
    {
        var               hits   = 0;
        Span<NumberTypeT> output = stackalloc NumberTypeT[2];

        foreach (TrainingDataPoint<NumberTypeT> point in checkDataPoints)
        {
            neuralNetwork.RunInference(inferenceBuffer, point.Inputs.Span, output);
            // ReSharper disable once CompareOfFloatsByEqualityOperator (These comparisons don't have that risk)
            if (IsSafe(point.ExpectedOutputs.Span))
            {
                if (IsSafeish(output)) hits++;
            }
            else
            {
                if (IsUnsafeish(output)) hits++;
            }
        }

        return (double) hits / checkDataPoints.Length;
    }

    // ReSharper disable once CompareOfFloatsByEqualityOperator
    public static bool IsSafe(this ReadOnlySpan<NumberTypeT> outputs) => outputs[0] == 1 && outputs[1] == 0;

    public static bool IsSafeish(this ReadOnlySpan<NumberTypeT> outputs) => outputs[0] > outputs[1];

    public static bool IsUnsafeish(this ReadOnlySpan<NumberTypeT> outputs) => outputs[0] < outputs[1];
}
