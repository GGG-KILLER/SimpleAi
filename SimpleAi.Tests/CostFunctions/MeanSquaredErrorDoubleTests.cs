using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi.Tests.CostFunctions;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class MeanSquaredErrorDoubleTests
{
    [Theory, MemberData(nameof(CostFunctionsTestData.InputSizes), MemberType = typeof(CostFunctionsTestData))]
    public void MeanSquaredErrorX2ECalculate_Returns_correct_values_for_any_input_size_with_doubles(int inputSize)
    {
        Tensor<double> costExpectedOutputs = Tensor.Create<double>([inputSize]);
        Tensor<double> costActualOutputs   = Tensor.Create<double>([inputSize]);
        for (var iteration = 0; iteration < CostFunctionsTestData.IterationCount; iteration++)
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

            Assert.Equal(expected, actual, CostFunctionsTestData.MeanSquaredErrorTolerance);
        }
    }
}
