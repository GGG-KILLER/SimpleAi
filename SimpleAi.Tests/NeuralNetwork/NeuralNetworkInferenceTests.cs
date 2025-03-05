using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi.Tests.NeuralNetwork;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class NeuralNetworkInferenceTests
{
    [Fact]
    public void NeuralNetworkX2ERunInference_Throws_exception_when_input_size_is_wrong()
    {
        var network = new NeuralNetwork<float>(new Layer<float, ReLu<float>>(inputs: 2, neurons: 2));

        Assert.Throws<ArgumentException>(() => network.RunInference((float[]) []));
        Assert.Throws<ArgumentException>(() => network.RunInference((float[]) [1]));
        Assert.Throws<ArgumentException>(() => network.RunInference((float[]) [1, 1, 1]));
    }

    [Fact]
    public void
        NeuralNetworkX2ERunInference_Calls_layers_in_correct_order_with_correct_inputs_on_an_odd_number_of_layers()
    {
        var network = new NeuralNetwork<float>(
            new Layer<float, ReLu<float>>(Tensor.Create([1f, 2f, 3f, 4f], [2, 2]), (float[]) [0f, 0f]),
            new Layer<float, ReLu<float>>(Tensor.Create([5f, 6f, 7f, 8f], [2, 2]), (float[]) [0f, 0f]),
            new Layer<float, ReLu<float>>(Tensor.Create([9f, 10f, 11f, 12f], [2, 2]), (float[]) [0f, 0f]));

        var output = network.RunInference((float[]) [13f, 14f]);

        // First hidden layer
        float a11 = ReLu<float>.Activate((float[]) [(1f * 13f) + (2f * 14f) + 0f])[0];
        float a12 = ReLu<float>.Activate((float[]) [(3f * 13f) + (4f * 14f) + 0f])[0];

        // Second hidden layer
        float a21 = ReLu<float>.Activate((float[]) [(5f * a11) + (6f * a12) + 0f])[0];
        float a22 = ReLu<float>.Activate((float[]) [(7f * a11) + (8f * a12) + 0f])[0];

        // Output Layer
        float a31 = ReLu<float>.Activate((float[]) [(9f * a21) + (10f * a22) + 0f])[0];
        float a32 = ReLu<float>.Activate((float[]) [(11f * a21) + (12f * a22) + 0f])[0];

        Assert.Equal(a31, output[0]);
        Assert.Equal(a32, output[1]);
    }

    [Fact]
    public void
        NeuralNetworkX2ERunInference_Calls_layers_in_correct_order_with_correct_inputs_on_an_even_number_of_layers()
    {
        var network = new NeuralNetwork<float>(
            new Layer<float, ReLu<float>>(Tensor.Create([1f, 2f, 3f, 4f], [2, 2]), (float[]) [0f, 0f]),
            new Layer<float, ReLu<float>>(Tensor.Create([5f, 6f, 7f, 8f], [2, 2]), (float[]) [0f, 0f]));

        var output = network.RunInference((float[]) [13f, 14f]);

        // First hidden layer
        float a11 = ReLu<float>.Activate((float[]) [(1f * 13f) + (2f * 14f) + 0f])[0];
        float a12 = ReLu<float>.Activate((float[]) [(3f * 13f) + (4f * 14f) + 0f])[0];

        // Output Layer
        float a21 = ReLu<float>.Activate((float[]) [(5f * a11) + (6f * a12) + 0f])[0];
        float a22 = ReLu<float>.Activate((float[]) [(7f * a11) + (8f * a12) + 0f])[0];

        Assert.Equal(a21, output[0]);
        Assert.Equal(a22, output[1]);
    }
}
