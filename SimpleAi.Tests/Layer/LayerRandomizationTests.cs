namespace SimpleAi.Tests.Layer;

public class LayerRandomizationTests
{
    [Fact]
    public void LayerX2ERandomize_Actually_randomizes_weights_and_biases()
    {
        var layer = new Layer<double, ReLU<double>, MeanSquaredError<double>>(2, 5);

        layer.Randomize(1.5);
        layer.Randomize(2.5);
        layer.Randomize(5.0);

        var weights = LayerAccessors.GetWeights(layer);
        var biases = LayerAccessors.GetBiases(layer);

        Assert.Contains(weights, x => x != 0.0);
        Assert.Contains(biases, x => x != 0.0);
    }

    [Fact]
    public void LayerX2ERandomize_Actually_randomizes_weights_and_biases_with_integers()
    {
        var layer = new Layer<long, ReLU<long>, MeanSquaredError<long>>(2, 5);

        layer.Randomize(2);
        layer.Randomize(3);
        layer.Randomize(5);

        var weights = LayerAccessors.GetWeights(layer);
        var biases = LayerAccessors.GetBiases(layer);

        Assert.Contains(weights, x => x != 0.0);
        Assert.Contains(biases, x => x != 0.0);
    }
}
