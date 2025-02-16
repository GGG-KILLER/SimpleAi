using JetBrains.Annotations;

namespace SimpleAi.Tests.NeuralNetwork;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class NeuralNetworkConstructorTests
{
    [Fact]
    public void NeuralNetworkX2Ector_Throws_exception_on_no_layers()
    {
        Assert.Throws<ArgumentException>(static () => new NeuralNetwork<float>());
    }

    [Fact]
    public void NeuralNetworkX2Ector_Throws_exception_on_mismatched_layers()
    {
        Assert.Throws<ArgumentException>(
            static () => new NeuralNetwork<float>(
                new Layer<float, ReLU<float>>(2, 2),
                new Layer<float, ReLU<float>>(4, 2)));
    }

    [Fact]
    public void NeuralNetworkX2Ector_Creates_correctly_assign_inputs_outputs_and_layers()
    {
        var network = new NeuralNetwork<double>(
            new Layer<double, ReLU<double>>(2, 10),
            new Layer<double, ReLU<double>>(10, 15));

        Assert.Equal(expected: 2, network.Inputs);
        Assert.Equal(expected: 15, network.Outputs);

        Assert.Equal(expected: 2, network.Layers.Length);
        Assert.Equal(expected: 2, network.Layers[index: 0].Inputs);
        Assert.Equal(expected: 10, network.Layers[index: 0].Outputs);
        Assert.Equal(expected: 10, network.Layers[index: 1].Inputs);
        Assert.Equal(expected: 15, network.Layers[index: 1].Outputs);
    }
}
