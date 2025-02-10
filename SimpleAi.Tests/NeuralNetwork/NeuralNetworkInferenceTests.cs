namespace SimpleAi.Tests.NeuralNetwork;

public class NeuralNetworkInferenceTests
{
    [Fact]
    public void NeuralNetworkX2ERunInference_Throws_exception_when_input_size_is_wrong()
    {
        var network = new NeuralNetwork<float, ReLU<float>, MeanSquaredError<float>>(2, 2);

        Assert.Throws<ArgumentException>(() =>
            network.RunInference([], stackalloc float[2]));
        Assert.Throws<ArgumentException>(() =>
            network.RunInference([1], stackalloc float[2]));
        Assert.Throws<ArgumentException>(() =>
            network.RunInference([1, 1, 1], stackalloc float[2]));
    }

    [Fact]
    public void NeuralNetworkX2ERunInference_Throws_exception_when_output_size_is_wrong()
    {
        var network = new NeuralNetwork<float, ReLU<float>, MeanSquaredError<float>>(2, 2);

        Assert.Throws<ArgumentException>(() =>
            network.RunInference([1, 1], []));
        Assert.Throws<ArgumentException>(() =>
            network.RunInference([1, 1], [1]));
        Assert.Throws<ArgumentException>(() =>
            network.RunInference([1, 1], [1, 1, 1]));
    }

    [Fact]
    public void NeuralNetworkX2ERunInference_Calls_layers_in_correct_order_with_correct_inputs_on_an_odd_number_of_layers()
    {
        var network = NeuralNetwork<float, ReLU<float>, MeanSquaredError<float>>.UnsafeLoad([
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

        Span<float> output = stackalloc float[2];
        network.RunInference([13, 14], output);

        // First hidden layer
        var a11 = ActivationHelper.Activate<float, ReLU<float>>(1f * 13f + 2f * 14f + 0f);
        var a12 = ActivationHelper.Activate<float, ReLU<float>>(3f * 13f + 4f * 14f + 0f);

        // Second hidden layer
        var a21 = ActivationHelper.Activate<float, ReLU<float>>(5f * a11 + 6f * a12 + 0f);
        var a22 = ActivationHelper.Activate<float, ReLU<float>>(7f * a11 + 8f * a12 + 0f);

        // Output Layer
        var a31 = ActivationHelper.Activate<float, ReLU<float>>(9f * a21 + 10f * a22 + 0f);
        var a32 = ActivationHelper.Activate<float, ReLU<float>>(11f * a21 + 12f * a22 + 0f);

        Assert.Equal(a31, output[0]);
        Assert.Equal(a32, output[1]);
    }

    [Fact]
    public void NeuralNetworkX2ERunInference_Calls_layers_in_correct_order_with_correct_inputs_on_an_even_number_of_layers()
    {
        var network = NeuralNetwork<float, ReLU<float>, MeanSquaredError<float>>.UnsafeLoad([
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

        Span<float> output = stackalloc float[2];
        network.RunInference([13, 14], output);

        // First hidden layer
        var a11 = ActivationHelper.Activate<float, ReLU<float>>(1f * 13f + 2f * 14f + 0f);
        var a12 = ActivationHelper.Activate<float, ReLU<float>>(3f * 13f + 4f * 14f + 0f);

        // Output Layer
        var a21 = ActivationHelper.Activate<float, ReLU<float>>(5f * a11 + 6f * a12 + 0f);
        var a22 = ActivationHelper.Activate<float, ReLU<float>>(7f * a11 + 8f * a12 + 0f);

        Assert.Equal(a21, output[0]);
        Assert.Equal(a22, output[1]);
    }
}
