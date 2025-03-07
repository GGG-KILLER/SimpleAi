using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi.Tests.LossFunctions;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class MeanSquaredErrorDoubleTests
{
    [Theory, MemberData(nameof(LossFunctionsTestData.InputSizes), MemberType = typeof(LossFunctionsTestData))]
    public void MeanSquaredErrorX2ECalculate_Returns_correct_values_for_any_input_size_with_doubles(int inputSize)
    {
        Tensor<double> costExpectedOutputs = Tensor.Create<double>([inputSize]);
        Tensor<double> costActualOutputs   = Tensor.Create<double>([inputSize]);
        for (var iteration = 0; iteration < LossFunctionsTestData.IterationCount; iteration++)
        {
            double expected = 0;
            for (var idx = 0; idx < inputSize; idx++)
            {
                costActualOutputs[idx]   = Random.Shared.NextDouble();
                costExpectedOutputs[idx] = Random.Shared.NextDouble();

                double error = costActualOutputs[idx] - costExpectedOutputs[idx];
                expected += error * error;
            }
            expected /= inputSize;

            double actual = MeanSquaredError<double>.Calculate(costExpectedOutputs, costActualOutputs);

            Assert.Equal(expected, actual, LossFunctionsTestData.MeanSquaredErrorTolerance);
        }
    }
}
