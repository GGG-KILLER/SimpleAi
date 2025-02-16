using JetBrains.Annotations;

namespace SimpleAi.Tests.ActivationFunctions;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class SigmoidFloatTests
{
    [Theory]
    [MemberData(nameof(ActivationFunctionsTestData.InputSizes), MemberType = typeof(ActivationFunctionsTestData))]
    public void SigmoidX2EActivate_Returns_expected_results_for_any_input_size_using_floats(int inputSize)
    {
        Span<float> expected = stackalloc float[inputSize];
        Span<float> inputs   = stackalloc float[inputSize];
        Span<float> outputs  = stackalloc float[inputSize];
        for (float n = -100f; n <= 100f; n += 0.125f)
        {
            inputs.Fill(n);
            expected.Fill(1 / (1 + float.Exp(-n)));
            Sigmoid<float>.Activate(inputs, outputs);
            for (var i = 0; i < outputs.Length; i++)
            {
                Assert.Equal(expected[i], outputs[i], 0.00001f);
            }
        }
    }
}
