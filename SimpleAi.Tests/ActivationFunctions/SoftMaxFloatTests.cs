using JetBrains.Annotations;

namespace SimpleAi.Tests.ActivationFunctions;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class SoftMaxFloatTests
{
    [Theory]
    [MemberData(nameof(ActivationFunctionsTestData.InputSizes), MemberType = typeof(ActivationFunctionsTestData))]
    public void SoftMaxX2EActivate_Returns_expected_results_for_any_input_size_using_floats(int inputSize)
    {
        Span<float> expected = stackalloc float[inputSize];
        Span<float> inputs   = stackalloc float[inputSize];
        Span<float> outputs  = stackalloc float[inputSize];
        for (float n = -100.0f; n <= 100.0f; n += 0.125f)
        {
            float expSum = 0;
            for (var idx = 0; idx < inputSize; idx++)
            {
                inputs[idx] =  Random.Shared.NextSingle();
                expSum      += float.Exp(inputs[idx]);
            }
            for (var idx = 0; idx < inputSize; idx++)
            {
                expected[idx] = float.Exp(inputs[idx]) / expSum;
            }

            SoftMax<float>.Activate(inputs, outputs);

            for (var i = 0; i < outputs.Length; i++)
            {
                Assert.Equal(expected[i], outputs[i], 0.00001);
            }
        }
    }
}
