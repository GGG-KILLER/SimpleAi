using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi.Tests.ActivationFunctions;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class SoftMaxDoubleTests
{
    [Theory,
     MemberData(nameof(ActivationFunctionsTestData.InputSizes), MemberType = typeof(ActivationFunctionsTestData))]
    public void SoftMaxX2EActivate_Returns_expected_results_for_any_input_size_using_doubles(int inputSize)
    {
        Tensor<double> expected = Tensor.Create<double>([inputSize]);
        Tensor<double> inputs   = Tensor.Create<double>([inputSize]);
        for (double n = -100.0; n <= 100.0; n += 0.125)
        {
            double expSum = 0;
            for (var idx = 0; idx < inputSize; idx++)
            {
                inputs[idx] =  Random.Shared.NextDouble();
                expSum      += double.Exp(inputs[idx]);
            }
            for (var idx = 0; idx < inputSize; idx++) expected[idx] = double.Exp(inputs[idx]) / expSum;

            var outputs = Softmax<double>.Activate(inputs);

            for (var i = 0; i < inputSize; i++) Assert.Equal(expected[i], outputs[i], tolerance: 0.00001);
        }
    }
}
