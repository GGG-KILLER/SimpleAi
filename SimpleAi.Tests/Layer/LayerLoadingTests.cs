namespace SimpleAi.Tests.Layer;

public class LayerLoadingTests
{
    [Fact]
    public void Layer_LoadUnsafe_CorrectlyCopiesValuesIntoFields()
    {
        var layer = Layer<ulong, ReLU<ulong>>.LoadUnsafe([1UL, 2UL, 3UL, 4UL, 5UL, 6UL], [7UL, 8UL]);

        Assert.Equal(3, layer.Inputs);
        Assert.Equal(2, layer.Size);

        Assert.Equal([1UL, 2UL, 3UL, 4UL, 5UL, 6UL], LayerAccessors.GetWeights(layer));
        Assert.Equal([7Ul, 8UL], LayerAccessors.GetBiases(layer));
    }
}
