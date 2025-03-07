using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi.Tests.LossFunctions;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class MeanSquaredErrorFloatTests
{
    [Theory, MemberData(nameof(LossFunctionsTestData.InputSizes), MemberType = typeof(LossFunctionsTestData))]
    public void MeanSquaredErrorX2ECalculate_Returns_correct_values_for_any_input_size_with_floats(int inputSize)
    {
        Tensor<float> costExpectedOutputs = Tensor.Create<float>([inputSize]);
        Tensor<float> costActualOutputs   = Tensor.Create<float>([inputSize]);
        for (var iteration = 0; iteration < LossFunctionsTestData.IterationCount; iteration++)
        {
            float expected = 0;
            for (var idx = 0; idx < inputSize; idx++)
            {
                costActualOutputs[idx]   = Random.Shared.NextSingle();
                costExpectedOutputs[idx] = Random.Shared.NextSingle();

                float error = costActualOutputs[idx] - costExpectedOutputs[idx];
                expected += error * error;
            }
            expected /= inputSize;

            float actual = MeanSquaredError<float>.Calculate(costActualOutputs, costExpectedOutputs);

            Assert.Equal(expected, actual, LossFunctionsTestData.MeanSquaredErrorTolerance);
        }
    }
}
