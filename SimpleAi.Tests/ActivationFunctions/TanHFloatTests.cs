using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi.Tests.ActivationFunctions;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class TanHFloatTests
{
    [Theory,
     MemberData(nameof(ActivationFunctionsTestData.InputSizes), MemberType = typeof(ActivationFunctionsTestData))]
    public void TanHX2EActivate_Returns_expected_results_for_any_input_size_using_floats(int inputSize)
    {
        Tensor<float> expected = Tensor.Create<float>([inputSize]);
        Tensor<float> inputs   = Tensor.Create<float>([inputSize]);
        for (float n = -100.0f; n <= 100.0f; n += 0.125f)
        {
            inputs.Fill(n);
            float e2 = float.Exp(2 * n);
            expected.Fill((e2 - 1) / (e2 + 1));
            var outputs = TanH<float>.Activate(inputs);
            for (var i = 0; i < inputSize; i++) Assert.Equal(expected[i], outputs[i], tolerance: 0.00001f);
        }
    }
}
