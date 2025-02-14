namespace SimpleAi.Tests.Layer;

public class LayerLoadingTests
{
    [Fact]
    public void LayerX2ELoadUnsafe_Correctly_copies_values_into_fields()
    {
        Layer<ulong, ReLU<ulong>>? layer = Layer<ulong, ReLU<ulong>>.LoadUnsafe(
            [1UL, 2UL, 3UL, 4UL, 5UL, 6UL],
            [7UL, 8UL]);

        Assert.Equal(expected: 3, layer.Inputs);
        Assert.Equal(expected: 2, layer.Size);

        Assert.Equal([1UL, 2UL, 3UL, 4UL, 5UL, 6UL], LayerAccessors.GetWeights(layer));
        Assert.Equal([7Ul, 8UL], LayerAccessors.GetBiases(layer));
    }
}
