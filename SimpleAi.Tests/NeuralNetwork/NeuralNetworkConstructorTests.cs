namespace SimpleAi.Tests.NeuralNetwork;

public class NeuralNetworkConstructorTests
{
    [Fact]
    public void NeuralNetworkX2Ector_Throws_exception_on_negative_or_zero_input_count()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuralNetwork<float, ReLU<float>>(inputs: -1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuralNetwork<float, ReLU<float>>(inputs: 0, 1));
    }

    [Fact]
    public void NeuralNetworkX2Ector_Throws_exception_without_any_layer_sizes()
    {
        Assert.Throws<ArgumentException>(() => new NeuralNetwork<float, ReLU<float>>(inputs: 1));
    }

    [Fact]
    public void NeuralNetworkX2Ector_Throws_exception_with_negative_or_zeroed_layer_sizes()
    {
        Assert.Throws<ArgumentException>(() => new NeuralNetwork<float, ReLU<float>>(inputs: 1, -1));
        Assert.Throws<ArgumentException>(() => new NeuralNetwork<float, ReLU<float>>(inputs: 1, 0));
    }

    [Fact]
    public void NeuralNetworkX2Ector_Creates_correct_layer_structure()
    {
        var network = new NeuralNetwork<double, ReLU<double>>(inputs: 2, 5, 5, 3);

        Assert.Equal(expected: 2, network.Inputs);
        Assert.Equal(expected: 3, network.Outputs);

        Assert.Equal(expected: 3, network.Layers.Length);
        Assert.Equal(expected: 2, network.Layers[index: 0].Inputs);
        Assert.Equal(expected: 5, network.Layers[index: 0].Size);
        Assert.Equal(expected: 5, network.Layers[index: 1].Inputs);
        Assert.Equal(expected: 5, network.Layers[index: 1].Size);
        Assert.Equal(expected: 5, network.Layers[index: 2].Inputs);
        Assert.Equal(expected: 3, network.Layers[index: 2].Size);
    }
}
