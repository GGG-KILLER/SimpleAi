namespace SimpleAi.Tests.Layer;

public class LayerConstructorTests
{
    [Fact]
    public void LayerX2Ector_Throws_on_negative_input_count()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<byte, ReLU<byte>, MeanSquaredError<byte>>(-1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<float, ReLU<float>, MeanSquaredError<float>>(-1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<double, ReLU<double>, MeanSquaredError<double>>(-1, 1));
    }

    [Fact]
    public void LayerX2Ector_Throws_on_zero_input_count()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<byte, ReLU<byte>, MeanSquaredError<byte>>(0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<float, ReLU<float>, MeanSquaredError<float>>(0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<double, ReLU<double>, MeanSquaredError<double>>(0, 1));
    }

    [Fact]
    public void LayerX2Ector_Throws_on_negative_size()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<byte, ReLU<byte>, MeanSquaredError<byte>>(1, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<float, ReLU<float>, MeanSquaredError<float>>(1, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<double, ReLU<double>, MeanSquaredError<double>>(1, -1));
    }

    [Fact]
    public void LayerX2Ector_Throws_on_zero_size()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<byte, ReLU<byte>, MeanSquaredError<byte>>(1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<float, ReLU<float>, MeanSquaredError<float>>(1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Layer<double, ReLU<double>, MeanSquaredError<double>>(1, 0));
    }
}
