using JetBrains.Annotations;

namespace SimpleAi.Tests.ActivationFunctions;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class TanHDoubleTests
{
    [Theory]
    [MemberData(nameof(ActivationFunctionsTestData.InputSizes), MemberType = typeof(ActivationFunctionsTestData))]
    public void TanHX2EActivate_Returns_expected_results_for_any_input_size_using_doubles(int inputSize)
    {
        Span<double> expected = stackalloc double[inputSize];
        Span<double> inputs   = stackalloc double[inputSize];
        Span<double> outputs  = stackalloc double[inputSize];
        for (double n = -100.0; n <= 100.0; n += 0.125)
        {
            inputs.Fill(n);
            double e2 = double.Exp(2 * n);
            expected.Fill((e2 - 1) / (e2 + 1));
            TanH<double>.Activate(inputs, outputs);
            for (var i = 0; i < outputs.Length; i++)
            {
                Assert.Equal(expected[i], outputs[i], 0.00001);
            }
        }
    }
}
