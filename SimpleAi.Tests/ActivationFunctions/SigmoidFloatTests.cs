using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi.Tests.ActivationFunctions;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class SigmoidFloatTests
{
    [Theory,
     MemberData(nameof(ActivationFunctionsTestData.InputSizes), MemberType = typeof(ActivationFunctionsTestData))]
    public void SigmoidX2EActivate_Returns_expected_results_for_any_input_size_using_floats(int inputSize)
    {
        Tensor<float> expected = Tensor.Create<float>([inputSize]);
        Tensor<float> inputs   = Tensor.Create<float>([inputSize]);
        for (float n = -100f; n <= 100f; n += 0.125f)
        {
            inputs.Fill(n);
            expected.Fill(1 / (1 + float.Exp(-n)));
            var outputs = Sigmoid<float>.Activate(inputs);
            for (var i = 0; i < inputSize; i++) Assert.Equal(expected[i], outputs[i], tolerance: 0.00001f);
        }
    }
}
