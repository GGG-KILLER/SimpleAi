namespace SimpleAi.Tests.NeuralNetwork;

public class NeuralNetworkConstructorTests
{
    [Fact]
    public void NeuralNetworkX2Ector_Throws_exception_on_negative_or_zero_input_count()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuralNetwork<float, ReLU<float>, MeanSquaredError<float>>(-1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuralNetwork<float, ReLU<float>, MeanSquaredError<float>>(0, 1));
    }

    [Fact]
    public void NeuralNetworkX2Ector_Throws_exception_without_any_layer_sizes()
    {
        Assert.Throws<ArgumentException>(() => new NeuralNetwork<float, ReLU<float>, MeanSquaredError<float>>(1));
    }

    [Fact]
    public void NeuralNetworkX2Ector_Throws_exception_with_negative_or_zeroed_layer_sizes()
    {
        Assert.Throws<ArgumentException>(() => new NeuralNetwork<float, ReLU<float>, MeanSquaredError<float>>(1, -1));
        Assert.Throws<ArgumentException>(() => new NeuralNetwork<float, ReLU<float>, MeanSquaredError<float>>(1, 0));
    }

    [Fact]
    public void NeuralNetworkX2Ector_Creates_correct_layer_structure()
    {
        var network = new NeuralNetwork<double, ReLU<double>, MeanSquaredError<double>>(2, 5, 5, 3);

        Assert.Equal(2, network.Inputs);
        Assert.Equal(3, network.Outputs);

        Assert.Equal(3, network.Layers.Length);
        Assert.Equal(2, network.Layers[0].Inputs);
        Assert.Equal(5, network.Layers[0].Size);
        Assert.Equal(5, network.Layers[1].Inputs);
        Assert.Equal(5, network.Layers[1].Size);
        Assert.Equal(5, network.Layers[2].Inputs);
        Assert.Equal(3, network.Layers[2].Size);
    }
}
