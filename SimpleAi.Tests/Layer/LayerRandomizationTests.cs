namespace SimpleAi.Tests.Layer;

public class LayerRandomizationTests
{
    [Fact]
    public void Layer_Randomize_ActuallyRandomizesWeightsAndBiases()
    {
        var layer = new Layer<double, ReLU<double>>(2, 5);

        layer.Randomize(1.5);
        layer.Randomize(2.5);
        layer.Randomize(5.0);

        var weights = LayerAccessors.GetWeights(layer);
        var biases = LayerAccessors.GetBiases(layer);

        Assert.Contains(weights, x => x != 0.0);
        Assert.Contains(biases, x => x != 0.0);
    }

    [Fact]
    public void Layer_Randomize_ActuallyRandomizesWeightsAndBiasesWithIntegers()
    {
        var layer = new Layer<long, ReLU<long>>(2, 5);

        layer.Randomize(2);
        layer.Randomize(3);
        layer.Randomize(5);

        var weights = LayerAccessors.GetWeights(layer);
        var biases = LayerAccessors.GetBiases(layer);

        Assert.Contains(weights, x => x != 0.0);
        Assert.Contains(biases, x => x != 0.0);
    }
}
