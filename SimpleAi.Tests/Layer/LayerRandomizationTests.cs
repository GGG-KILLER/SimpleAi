using JetBrains.Annotations;

namespace SimpleAi.Tests.Layer;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class LayerRandomizationTests
{
    [Fact]
    public void LayerX2ERandomizeWeights_Properly_randomizes_weights()
    {
        var layer = new Layer<double, ReLU<double>>(inputCount: 2, size: 5);

        layer.RandomizeWeights(mean: 25, stdDev: 5);

        double[] weights = LayerAccessors.GetWeights(layer);

        Assert.Contains(weights, x => x is >= 20 and <= 30);
    }

    [Fact]
    public void LayerX2ERandomizeWeights_Properly_randomizes_weights_with_integers()
    {
        var layer = new Layer<long, ReLU<long>>(inputCount: 2, size: 5);

        layer.RandomizeWeights(mean: 25, stdDev: 5);

        long[] weights = LayerAccessors.GetWeights(layer);

        Assert.Contains(weights, x => x is >= 20 and <= 30);
    }
}
