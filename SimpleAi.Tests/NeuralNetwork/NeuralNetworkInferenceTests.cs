using JetBrains.Annotations;

namespace SimpleAi.Tests.NeuralNetwork;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class NeuralNetworkInferenceTests
{
    [Fact]
    public void NeuralNetworkX2ERunInference_Throws_exception_when_input_size_is_wrong()
    {
        var network = new NeuralNetwork<float, ReLU<float>>(inputs: 2, 2);
        var session = new InferenceSession<float>(network);

        Assert.Throws<ArgumentException>(() => network.RunInference(session, [], stackalloc float[2]));
        Assert.Throws<ArgumentException>(() => network.RunInference(session, [1], stackalloc float[2]));
        Assert.Throws<ArgumentException>(() => network.RunInference(session, [1, 1, 1], stackalloc float[2]));
    }

    [Fact]
    public void NeuralNetworkX2ERunInference_Throws_exception_when_output_size_is_wrong()
    {
        var network = new NeuralNetwork<float, ReLU<float>>(inputs: 2, 2);
        var session = new InferenceSession<float>(network);

        Assert.Throws<ArgumentException>(() => network.RunInference(session, [1, 1], []));
        Assert.Throws<ArgumentException>(() => network.RunInference(session, [1, 1], [1]));
        Assert.Throws<ArgumentException>(() => network.RunInference(session, [1, 1], [1, 1, 1]));
    }

    [Fact]
    public void
        NeuralNetworkX2ERunInference_Calls_layers_in_correct_order_with_correct_inputs_on_an_odd_number_of_layers()
    {
        // @formatter:off
        NeuralNetwork<float, ReLU<float>> network = NeuralNetwork<float, ReLU<float>>.UnsafeLoad([
            [
                1f, 2f,
                3f, 4f,
            ],
            [
                5f, 6f,
                7f, 8f,
            ],
            [
                 9f, 10f,
                11f, 12f,
            ],
        ], [
            [0f, 0f],
            [0f, 0f],
            [0f, 0f],
        ]);
        // @formatter:on
        var session = new InferenceSession<float>(network);

        Span<float> output = stackalloc float[2];
        network.RunInference(session, [13, 14], output);

        // First hidden layer
        float a11 = ActivationHelper.Activate<float, ReLU<float>>((1f * 13f) + (2f * 14f) + 0f);
        float a12 = ActivationHelper.Activate<float, ReLU<float>>((3f * 13f) + (4f * 14f) + 0f);

        // Second hidden layer
        float a21 = ActivationHelper.Activate<float, ReLU<float>>((5f * a11) + (6f * a12) + 0f);
        float a22 = ActivationHelper.Activate<float, ReLU<float>>((7f * a11) + (8f * a12) + 0f);

        // Output Layer
        float a31 = ActivationHelper.Activate<float, ReLU<float>>((9f * a21) + (10f * a22) + 0f);
        float a32 = ActivationHelper.Activate<float, ReLU<float>>((11f * a21) + (12f * a22) + 0f);

        Assert.Equal(a31, output[index: 0]);
        Assert.Equal(a32, output[index: 1]);
    }

    [Fact]
    public void
        NeuralNetworkX2ERunInference_Calls_layers_in_correct_order_with_correct_inputs_on_an_even_number_of_layers()
    {
        // @formatter:off
        NeuralNetwork<float, ReLU<float>> network = NeuralNetwork<float, ReLU<float>>.UnsafeLoad([
            [
                1f, 2f,
                3f, 4f,
            ],
            [
                5f, 6f,
                7f, 8f,
            ],
        ], [
            [0f, 0f],
            [0f, 0f],
        ]);
        // @formatter:on
        var session = new InferenceSession<float>(network);

        Span<float> output = stackalloc float[2];
        network.RunInference(session, [13, 14], output);

        // First hidden layer
        float a11 = ActivationHelper.Activate<float, ReLU<float>>((1f * 13f) + (2f * 14f) + 0f);
        float a12 = ActivationHelper.Activate<float, ReLU<float>>((3f * 13f) + (4f * 14f) + 0f);

        // Output Layer
        float a21 = ActivationHelper.Activate<float, ReLU<float>>((5f * a11) + (6f * a12) + 0f);
        float a22 = ActivationHelper.Activate<float, ReLU<float>>((7f * a11) + (8f * a12) + 0f);

        Assert.Equal(a21, output[index: 0]);
        Assert.Equal(a22, output[index: 1]);
    }
}
