using JetBrains.Annotations;

namespace SimpleAi.Tests.ActivationFunctions;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class SoftMaxDoubleTests
{
    [Theory]
    [MemberData(nameof(ActivationFunctionsTestData.InputSizes), MemberType = typeof(ActivationFunctionsTestData))]
    public void SoftMaxX2EActivate_Returns_expected_results_for_any_input_size_using_doubles(int inputSize)
    {
        Span<double> expected = stackalloc double[inputSize];
        Span<double> inputs   = stackalloc double[inputSize];
        Span<double> outputs  = stackalloc double[inputSize];
        for (double n = -100.0; n <= 100.0; n += 0.125)
        {
            double expSum = 0;
            for (var idx = 0; idx < inputSize; idx++)
            {
                inputs[idx] =  Random.Shared.NextDouble();
                expSum      += double.Exp(inputs[idx]);
            }
            for (var idx = 0; idx < inputSize; idx++)
            {
                expected[idx] = double.Exp(inputs[idx]) / expSum;
            }

            SoftMax<double>.Activate(inputs, outputs);

            for (var i = 0; i < outputs.Length; i++)
            {
                Assert.Equal(expected[i], outputs[i], 0.00001);
            }
        }
    }
}
