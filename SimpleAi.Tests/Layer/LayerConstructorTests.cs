using JetBrains.Annotations;

namespace SimpleAi.Tests.Layer;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class LayerConstructorTests
{
    [Fact]
    public void LayerX2Ector_Throws_on_negative_input_count()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<byte, ReLU<byte>>(inputCount: -1, size: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<float, ReLU<float>>(inputCount: -1, size: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<double, ReLU<double>>(inputCount: -1, size: 1));
    }

    [Fact]
    public void LayerX2Ector_Throws_on_zero_input_count()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<byte, ReLU<byte>>(inputCount: 0, size: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<float, ReLU<float>>(inputCount: 0, size: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<double, ReLU<double>>(inputCount: 0, size: 1));
    }

    [Fact]
    public void LayerX2Ector_Throws_on_negative_size()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<byte, ReLU<byte>>(inputCount: 1, size: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<float, ReLU<float>>(inputCount: 1, size: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<double, ReLU<double>>(inputCount: 1, size: -1));
    }

    [Fact]
    public void LayerX2Ector_Throws_on_zero_size()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<byte, ReLU<byte>>(inputCount: 1, size: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<float, ReLU<float>>(inputCount: 1, size: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<double, ReLU<double>>(inputCount: 1, size: 0));
    }
}
