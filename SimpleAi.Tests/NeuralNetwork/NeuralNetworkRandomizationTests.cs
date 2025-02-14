namespace SimpleAi.Tests.NeuralNetwork;

public class NeuralNetworkRandomizationTests
{
    [Fact]
    public void NeuralNetworkX2ERandomizeWeights_Properly_randomizes_layersX27_weights()
    {
        var network = new NeuralNetwork<long, ReLU<long>>(inputs: 2, 3);

        network.RandomizeWeights(mean: 25, stdDev: 5);

        foreach (Layer<long, ReLU<long>> layer in network.Layers)
        {
            long[] weights = LayerAccessors.GetWeights(layer);

            Assert.Contains(weights, x => 20 <= x && x <= 30);
        }
    }
}
