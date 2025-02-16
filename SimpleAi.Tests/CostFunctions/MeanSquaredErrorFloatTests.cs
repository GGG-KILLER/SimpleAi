using JetBrains.Annotations;

namespace SimpleAi.Tests.CostFunctions;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class MeanSquaredErrorFloatTests
{
    [Theory]
    [MemberData(nameof(CostFunctionsTestData.InputSizes), MemberType = typeof(CostFunctionsTestData))]
    public void MeanSquaredErrorX2ECalculate_Returns_correct_values_for_any_input_size_with_floats(int inputSize)
    {
        Span<float> costExpectedOutputs = stackalloc float[inputSize];
        Span<float> costActualOutputs   = stackalloc float[inputSize];
        for (var iteration = 0; iteration < CostFunctionsTestData.IterationCount; iteration++)
        {
            float expected = 0;
            for (var idx = 0; idx < inputSize; idx++)
            {
                costActualOutputs[idx]   = Random.Shared.NextSingle();
                costExpectedOutputs[idx] = Random.Shared.NextSingle();

                float error = costActualOutputs[idx] - costExpectedOutputs[idx];
                expected += error * error;
            }
            expected *= 0.5f;

            float actual = MeanSquaredError<float>.Calculate(costActualOutputs, costExpectedOutputs);

            Assert.Equal(expected, actual, CostFunctionsTestData.MeanSquaredErrorTolerance);
        }
    }
}
