using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi.Tests.ActivationFunctions;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class SoftMaxFloatTests
{
    [Theory,
     MemberData(nameof(ActivationFunctionsTestData.InputSizes), MemberType = typeof(ActivationFunctionsTestData))]
    public void SoftMaxX2EActivate_Returns_expected_results_for_any_input_size_using_floats(int inputSize)
    {
        Tensor<float> expected = Tensor.Create<float>([inputSize]);
        Tensor<float> inputs   = Tensor.Create<float>([inputSize]);
        for (float n = -100.0f; n <= 100.0f; n += 0.125f)
        {
            float expSum = 0;
            for (var idx = 0; idx < inputSize; idx++)
            {
                inputs[idx] =  Random.Shared.NextSingle();
                expSum      += float.Exp(inputs[idx]);
            }
            for (var idx = 0; idx < inputSize; idx++) expected[idx] = float.Exp(inputs[idx]) / expSum;

            var outputs = Softmax<float>.Activate(inputs);

            for (var i = 0; i < inputSize; i++) Assert.Equal(expected[i], outputs[i], tolerance: 0.00001);
        }
    }
}
