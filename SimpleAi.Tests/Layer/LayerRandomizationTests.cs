namespace SimpleAi.Tests.Layer;

public class LayerRandomizationTests
{
    [Fact]
    public void LayerX2ERandomizeWeights_Properly_randomizes_weights()
    {
        var layer = new Layer<double, ReLU<double>, MeanSquaredError<double>>(2, 5);

        layer.RandomizeWeights(25, 5);

        var weights = LayerAccessors.GetWeights(layer);

        Assert.Contains(weights, x => 20 <= x && x <= 30);
    }

    [Fact]
    public void LayerX2ERandomizeWeights_Properly_randomizes_weights_with_integers()
    {
        var layer = new Layer<long, ReLU<long>, MeanSquaredError<long>>(2, 5);

        layer.RandomizeWeights(25, 5);

        var weights = LayerAccessors.GetWeights(layer);

        Assert.Contains(weights, x => 20 <= x && x <= 30);
    }
}
