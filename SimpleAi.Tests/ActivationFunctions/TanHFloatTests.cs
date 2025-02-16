namespace SimpleAi.Tests.ActivationFunctions;

public class TanHFloatTests
{
    [Theory]
    [MemberData(nameof(ActivationFunctionsTestData.InputSizes), MemberType = typeof(ActivationFunctionsTestData))]
    public void TanHX2EActivate_Returns_expected_results_for_any_input_size_using_floats(int inputSize)
    {
        Span<float> expected = stackalloc float[inputSize];
        Span<float> inputs   = stackalloc float[inputSize];
        Span<float> outputs  = stackalloc float[inputSize];
        for (float n = -100.0f; n <= 100.0f; n += 0.125f)
        {
            inputs.Fill(n);
            float e2 = float.Exp(2 * n);
            expected.Fill((e2 - 1) / (e2 + 1));
            TanH<float>.Activate(inputs, outputs);
            for (var i = 0; i < outputs.Length; i++)
            {
                Assert.Equal(expected[i], outputs[i], 0.00001f);
            }
        }
    }
}
