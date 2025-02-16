using JetBrains.Annotations;

namespace SimpleAi.Tests.Layer;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class LayerRandomizationTests
{
    [Fact(Skip = "TODO: Fix normal distribution")]
    public void LayerX2ERandomizeWeights_Properly_randomizes_weights()
    {
        var layer = new Layer<double, ReLU<double>>(inputCount: 2, outputCount: 5);

        layer.RandomizeWeights(mean: 25, stdDev: 5);

        Assert.True(layer.Weights.IndexOfAnyExceptInRange(20, 30) == -1);
    }

    [Fact(Skip = "TODO: Fix normal distribution")]
    public void LayerX2ERandomizeWeights_Properly_randomizes_weights_with_integers()
    {
        var layer = new Layer<long, ReLU<long>>(inputCount: 2, outputCount: 5);

        layer.RandomizeWeights(mean: 25, stdDev: 5);

        Assert.True(layer.Weights.IndexOfAnyExceptInRange(20, 30) == -1);
    }
}
