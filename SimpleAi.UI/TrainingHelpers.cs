using System;
using System.Linq;
using System.Numerics.Tensors;
using SimpleAi.UI.Maths;

namespace SimpleAi.UI;

internal static class TrainingHelpers
{
    private static NumberTypeT[] SafeOutput { get; } = [1, 0];

    private static NumberTypeT[] UnsafeOutput { get; } = [0, 1];

    public static TrainingDataPoint<NumberTypeT>[] GenerateTrainingData(
        Vector2DRange totalArea,
        Vector2DRange safeArea,
        int           unsafePoints,
        int           safePoints)
    {
        var trainingData    = new TrainingDataPoint<NumberTypeT>[safePoints + unsafePoints];
        var generatedSafe   = 0;
        var generatedUnsafe = 0;

        for (var idx = 0; idx < trainingData.Length; idx++)
        {
            bool isSafe;
            if (generatedSafe < safePoints && generatedUnsafe < unsafePoints)
                isSafe = Random.Shared.NextDouble() > 0.5;
            else
                isSafe = generatedSafe < safePoints;

            NumberTypeT[] inputs = new NumberTypeT[2];
            if (isSafe)
            {
                inputs[0] = (NumberTypeT) (safeArea.Start.X
                                           + (Random.Shared.NextDouble() * (safeArea.End.X - safeArea.Start.X)));
                inputs[1] = (NumberTypeT) (safeArea.Start.Y
                                           + (Random.Shared.NextDouble() * (safeArea.End.Y - safeArea.Start.Y)));
                generatedSafe++;
            }
            else
            {
            regen:
                inputs[0] = (NumberTypeT) (totalArea.Start.X
                                           + (Random.Shared.NextDouble() * (totalArea.End.X - totalArea.Start.X)));
                inputs[1] = (NumberTypeT) (totalArea.Start.Y
                                           + (Random.Shared.NextDouble() * (totalArea.End.Y - totalArea.Start.Y)));

                if (safeArea.Start.X < inputs[0]
                    && inputs[0] < safeArea.End.X
                    && safeArea.Start.Y < inputs[1]
                    && inputs[1] < safeArea.End.Y)
                    goto regen;

                generatedUnsafe++;
            }

            trainingData[idx] = new TrainingDataPoint<NumberTypeT>(inputs, isSafe ? SafeOutput : UnsafeOutput);
        }

        return trainingData;
    }

    public static double CalculateAccuracy(
        this NeuralNetwork<NumberTypeT> neuralNetwork,
        ITrainingData<NumberTypeT>      checkData)
    {
        int hits = checkData.AsParallel().Count(
            point => Tensor.IndexOfMax<NumberTypeT>(point.ExpectedOutputs)
                     == Tensor.IndexOfMax<NumberTypeT>(neuralNetwork.RunInference(point.Inputs)));
        return (double) hits / checkData.Length;
    }

    public static bool IsSafeish(in ReadOnlyTensorSpan<NumberTypeT> outputs) => outputs[0] > outputs[1];
}
