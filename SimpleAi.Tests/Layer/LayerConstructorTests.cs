namespace SimpleAi.Tests.Layer;

public class LayerConstructorTests
{
    [Fact]
    public void Layer_Constructor_ThrowsOnNegativeInputCount()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<byte, ReLU<byte>>(-1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<float, ReLU<float>>(-1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<double, ReLU<double>>(-1, 1));
    }

    [Fact]
    public void Layer_Constructor_ThrowsOnZeroInputCount()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<byte, ReLU<byte>>(0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<float, ReLU<float>>(0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<double, ReLU<double>>(0, 1));
    }

    [Fact]
    public void Layer_Constructor_ThrowsOnNegativeSize()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<byte, ReLU<byte>>(1, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<float, ReLU<float>>(1, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<double, ReLU<double>>(1, -1));
    }

    [Fact]
    public void Layer_Constructor_ThrowsOnZeroSize()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<byte, ReLU<byte>>(1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<float, ReLU<float>>(1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<double, ReLU<double>>(1, 0));
    }
}
