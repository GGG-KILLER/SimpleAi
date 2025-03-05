using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi.Tests.Layer;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class LayerConstructorTests
{
    [Fact]
    public void LayerX2Ector_Throws_on_negative_input_count()
    {
        Assert.Throws<ArgumentOutOfRangeException>(static () => new Layer<float, ReLu<float>>(inputs: -1, neurons: 1));
        Assert.Throws<ArgumentOutOfRangeException>(
            static () => new Layer<double, ReLu<double>>(inputs: -1, neurons: 1));
    }

    [Fact]
    public void LayerX2Ector_Throws_on_zero_input_count()
    {
        Assert.Throws<ArgumentOutOfRangeException>(static () => new Layer<float, ReLu<float>>(inputs: 0, neurons: 1));
        Assert.Throws<ArgumentOutOfRangeException>(static () => new Layer<double, ReLu<double>>(inputs: 0, neurons: 1));
    }

    [Fact]
    public void LayerX2Ector_Throws_on_negative_size()
    {
        Assert.Throws<ArgumentOutOfRangeException>(static () => new Layer<float, ReLu<float>>(inputs: 1, neurons: -1));
        Assert.Throws<ArgumentOutOfRangeException>(
            static () => new Layer<double, ReLu<double>>(inputs: 1, neurons: -1));
    }

    [Fact]
    public void LayerX2Ector_Throws_on_zero_size()
    {
        Assert.Throws<ArgumentOutOfRangeException>(static () => new Layer<float, ReLu<float>>(inputs: 1, neurons: 0));
        Assert.Throws<ArgumentOutOfRangeException>(static () => new Layer<double, ReLu<double>>(inputs: 1, neurons: 0));
    }

    [Fact]
    public void LayerX2Ector_Properly_initializes_layer_with_input_and_neuron_count()
    {
        var layer = new Layer<float, ReLu<float>>(inputs: 5, neurons: 2);

        Assert.Equal(5, layer.Inputs);
        Assert.Equal(2, layer.Neurons);

        Assert.Equal(2, layer.Weights.Rank);
        Assert.Equal([2, 5], layer.Weights.Lengths);

        Assert.Equal(1, layer.Biases.Rank);
        Assert.Equal([2], layer.Biases.Lengths);
    }

    [Fact]
    public void LayerX2Ector_Throws_on_non_2D_weights()
    {
        Assert.Throws<ArgumentException>(
            static () => new Layer<float, ReLu<float>>(Tensor.Create<float>([5]), Tensor.Create<float>([5])));
        Assert.Throws<ArgumentException>(
            static () => new Layer<double, ReLu<double>>(Tensor.Create<double>([5]), Tensor.Create<double>([5])));
    }

    [Fact]
    public void LayerX2Ector_Throws_on_non_1D_biases()
    {
        Assert.Throws<ArgumentException>(
            static () => new Layer<float, ReLu<float>>(Tensor.Create<float>([5, 2]), Tensor.Create<float>([2, 5])));
        Assert.Throws<ArgumentException>(
            static () => new Layer<double, ReLu<double>>(Tensor.Create<double>([5, 2]), Tensor.Create<double>([2, 5])));
    }

    [Fact]
    public void LayerX2Ector_Throws_on_empty_weights()
    {
        Assert.Throws<ArgumentException>(
            static () => new Layer<float, ReLu<float>>(Tensor.Create<float>([5, 2]), Tensor.Create<float>([2])));
        Assert.Throws<ArgumentException>(
            static () => new Layer<double, ReLu<double>>(Tensor.Create<double>([5, 2]), Tensor.Create<double>([2])));
    }

    [Fact]
    public void LayerX2Ector_Throws_on_empty_biases()
    {
        Assert.Throws<ArgumentException>(
            static () => new Layer<float, ReLu<float>>(
                Tensor.Create([1f, 2f, 3f, 4f, 6f, 7f, 8f, 9f, 10f], [5, 2]),
                Tensor.Create<float>([2, 5])));
        Assert.Throws<ArgumentException>(
            static () => new Layer<double, ReLu<double>>(
                Tensor.Create([1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0], [5, 2]),
                Tensor.Create<double>([2, 5])));
    }

    [Fact]
    public void LayerX2Ector_Throws_on_mismatched_weights_and_biases()
    {
        Assert.Throws<ArgumentException>(
            static () => new Layer<float, ReLu<float>>(
                Tensor.Create([1f, 2f, 3f, 4f, 6f, 7f, 8f, 9f, 10f], [5, 2]),
                Tensor.Create([1f, 2f, 3f, 4f, 5f], [5])));
        Assert.Throws<ArgumentException>(
            static () => new Layer<double, ReLu<double>>(
                Tensor.Create([1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0], [5, 2]),
                Tensor.Create([1.0, 2.0, 3.0, 4.0, 5.0], [5])));
    }

    [Fact]
    public void LayerX2Ector_Properly_stores_input_tensors()
    {
        var layer = new Layer<float, ReLu<float>>(
            Tensor.Create([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f], [2, 5]),
            Tensor.Create([11f, 12f], [2]));

        Assert.Equal(5, layer.Inputs);
        Assert.Equal(2, layer.Neurons);

        Assert.Equal(2, layer.Weights.Rank);
        Assert.Equal([2, 5], layer.Weights.Lengths);
        Assert.True(layer.Weights.SequenceEqual(Tensor.Create([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f], [2, 5])));

        Assert.Equal(1, layer.Biases.Rank);
        Assert.Equal([2], layer.Biases.Lengths);
        Assert.True(layer.Biases.SequenceEqual(Tensor.Create([11f, 12f], [2])));
    }
}
