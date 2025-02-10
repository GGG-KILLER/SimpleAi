namespace SimpleAi.Tests.NeuralNetwork;

public class NeuralNetworkRandomizationTests
{
    [Fact]
    public void NeuralNetworkX2ERandomize_Properly_randomizes_layers()
    {
        var network = new NeuralNetwork<long, ReLU<long>>(2, 3);

        network.Randomize(5);
        network.Randomize(10);
        network.Randomize(15);

        foreach (var layer in network.Layers)
        {
            var weights = LayerAccessors.GetWeights(layer);
            var biases = LayerAccessors.GetBiases(layer);

            Assert.Contains(weights, x => x != 0);
            Assert.Contains(biases, x => x != 0);
        }
    }
}
