using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi.Tests.ActivationFunctions;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
// ReSharper disable once InconsistentNaming
public class ReLUDoubleTests
{
    [Theory,
     MemberData(nameof(ActivationFunctionsTestData.InputSizes), MemberType = typeof(ActivationFunctionsTestData))]
    public void ReLUX2EActivate_Returns_expected_results_for_any_input_size_using_doubles(int inputSize)
    {
        Tensor<double> expected = Tensor.Create<double>([inputSize]);
        Tensor<double> inputs   = Tensor.Create<double>([inputSize]);
        for (double n = -100.0; n <= 100.0; n += 0.125)
        {
            inputs.Fill(n);
            expected.Fill(n > 0 ? n : 0);
            var outputs = ReLu<double>.Activate(inputs);
            for (nint i = 0; i < inputSize; i++) Assert.Equal(expected[i], outputs[i], tolerance: 0.00001);
        }
    }
}
