using JetBrains.Annotations;

namespace SimpleAi.Tests.CostFunctions;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class CrossEntropyFloatTests
{
    [Theory]
    [MemberData(nameof(CostFunctionsTestData.InputSizes), MemberType = typeof(CostFunctionsTestData))]
    public void CrossEntropyX2ECalculate_Returns_correct_values_for_any_input_size_with_floats(int inputSize)
    {
        Span<float> costExpectedOutputs = stackalloc float[inputSize];
        Span<float> costActualOutputs   = stackalloc float[inputSize];
        for (var iteration = 0; iteration < CostFunctionsTestData.IterationCount; iteration++)
        {
            float expected = 0;
            for (var idx = 0; idx < inputSize; idx++)
            {
                costActualOutputs[idx]   = Random.Shared.NextSingle();
                costExpectedOutputs[idx] = Random.Shared.NextSingle() >= 0.5f ? 1 : 0;

                // ReSharper disable once CompareOfFloatsByEqualityOperator
                float v = costExpectedOutputs[idx] == 1
                              ? -float.Log(costActualOutputs[idx])
                              : -float.Log(1 - costActualOutputs[idx]);
                expected += float.IsNaN(v) ? 0 : v;
            }

            float actual = CrossEntropy<float>.Calculate(costExpectedOutputs, costActualOutputs);

            Assert.Equal(expected, actual, CostFunctionsTestData.CrossEntropyTolerance);
        }
    }
}
