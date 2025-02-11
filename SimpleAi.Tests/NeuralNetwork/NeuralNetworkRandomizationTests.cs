namespace SimpleAi.Tests.NeuralNetwork;

public class NeuralNetworkRandomizationTests
{
    [Fact]
    public void NeuralNetworkX2ERandomizeWeights_Properly_randomizes_layersX27_weights()
    {
        var network = new NeuralNetwork<long, ReLU<long>>(2, 3);

        network.RandomizeWeights(25, 5);

        foreach (var layer in network.Layers)
        {
            var weights = LayerAccessors.GetWeights(layer);

            Assert.Contains(weights, x => 20 <= x && x <= 30);
        }
    }
}
