using JetBrains.Annotations;

namespace SimpleAi.Tests.NeuralNetwork;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class NeuralNetworkRandomizationTests
{
    [Fact(Skip = "TODO: Fix normal distribution")]
    public void NeuralNetworkX2ERandomizeWeights_Properly_randomizes_layersX27_weights()
    {
        var network = new NeuralNetwork<long>(new Layer<long, ReLU<long>>(2, 10));

        network.RandomizeWeights(mean: 25, stdDev: 5);

        foreach (var layer in network.Layers)
        {
            Assert.True(layer.Weights.AsSpan().IndexOfAnyExceptInRange(20, 30) == -1);
        }
    }
}
