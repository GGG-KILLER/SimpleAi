namespace SimpleAi.Tests.CostFunctions;

public class CrossEntropyDoubleTests
{
    [Theory]
    [MemberData(nameof(CostFunctionsTestData.InputSizes), MemberType = typeof(CostFunctionsTestData))]
    public void CrossEntropyX2ECalculate_Returns_correct_values_for_any_input_size_with_doubles(int inputSize)
    {
        Span<double> costExpectedOutputs = stackalloc double[inputSize];
        Span<double> costActualOutputs   = stackalloc double[inputSize];
        for (var iteration = 0; iteration < CostFunctionsTestData.IterationCount; iteration++)
        {
            double expected = 0;
            for (var idx = 0; idx < inputSize; idx++)
            {
                costActualOutputs[idx]   = Random.Shared.NextDouble();
                costExpectedOutputs[idx] = Random.Shared.NextDouble() >= 0.5 ? 1 : 0;

                // ReSharper disable once CompareOfFloatsByEqualityOperator
                double v = costExpectedOutputs[idx] == 1
                               ? -double.Log(costActualOutputs[idx])
                               : -double.Log(1 - costActualOutputs[idx]);
                expected += double.IsNaN(v) ? 0 : v;
            }

            double actual = CrossEntropy<double>.Calculate(costExpectedOutputs, costActualOutputs);

            Assert.Equal(expected, actual, CostFunctionsTestData.CrossEntropyTolerance);
        }
    }
}
