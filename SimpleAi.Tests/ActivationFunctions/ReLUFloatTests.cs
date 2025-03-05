using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi.Tests.ActivationFunctions;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
// ReSharper disable once InconsistentNaming
public class ReLUFloatTests
{
    [Theory,
     MemberData(nameof(ActivationFunctionsTestData.InputSizes), MemberType = typeof(ActivationFunctionsTestData))]
    public void ReLUX2EActivate_Returns_expected_results_for_any_input_size_using_floats(int inputSize)
    {
        Tensor<float> expected = Tensor.Create<float>([inputSize]);
        Tensor<float> inputs   = Tensor.Create<float>([inputSize]);
        for (float n = -100.0f; n <= 100.0f; n += 0.125f)
        {
            inputs.Fill(n);
            expected.Fill(n > 0 ? n : 0);
            var outputs = ReLu<float>.Activate(inputs);
            for (var i = 0; i < inputSize; i++) Assert.Equal(expected[i], outputs[i], tolerance: 0.00001);
        }
    }
}
