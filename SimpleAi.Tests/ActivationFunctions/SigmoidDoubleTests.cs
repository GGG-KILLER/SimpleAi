using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi.Tests.ActivationFunctions;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class SigmoidDoubleTests
{
    [Theory,
     MemberData(nameof(ActivationFunctionsTestData.InputSizes), MemberType = typeof(ActivationFunctionsTestData))]
    public void SigmoidX2EActivate_Returns_expected_results_for_any_input_size_using_doubles(int inputSize)
    {
        Tensor<double> expected = Tensor.Create<double>([inputSize]);
        Tensor<double> inputs   = Tensor.Create<double>([inputSize]);
        for (double n = -100.0; n <= 100.0; n += 0.125)
        {
            inputs.Fill(n);
            expected.Fill(1 / (1 + double.Exp(-n)));
            var outputs = Sigmoid<double>.Activate(inputs);
            for (var i = 0; i < inputSize; i++) Assert.Equal(expected[i], outputs[i], tolerance: 0.00001);
        }
    }
}
