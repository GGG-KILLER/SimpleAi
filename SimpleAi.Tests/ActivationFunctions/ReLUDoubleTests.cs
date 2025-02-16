namespace SimpleAi.Tests.ActivationFunctions;

// ReSharper disable once InconsistentNaming
public class ReLUDoubleTests
{
    [Theory]
    [MemberData(nameof(ActivationFunctionsTestData.InputSizes), MemberType = typeof(ActivationFunctionsTestData))]
    public void ReLUX2EActivate_Returns_expected_results_for_any_input_size_using_doubles(int inputSize)
    {
        Span<double> expected = stackalloc double[inputSize];
        Span<double> inputs   = stackalloc double[inputSize];
        Span<double> outputs  = stackalloc double[inputSize];
        for (double n = -100.0; n <= 100.0; n += 0.125)
        {
            inputs.Fill(n);
            expected.Fill(n > 0 ? n : 0);
            ReLU<double>.Activate(inputs, outputs);
            for (var i = 0; i < outputs.Length; i++)
            {
                Assert.Equal(expected[i], outputs[i], 0.00001);
            }
        }
    }
}
