namespace SimpleAi.Tests.NeuralNetwork;

public class NeuralNetworkLoadingTests
{
    [Fact]
    public void NeuralNetworkX2EUnsafeLoad_Correctly_sets_inputs_and_outputs_as_well_as_loads_layers()
    {
        var network1 = NeuralNetwork<float, ReLU<float>>.UnsafeLoad([
            [
                1f, 2f,
                3f, 4f,
            ],
            [
                5f, 6f,
                7f, 8f,
            ],
            [
                 9f, 10f,
                11f, 12f,
            ],
        ], [
            [0f, 0f],
            [0f, 0f],
            [0f, 0f],
        ]);

        var network2 = NeuralNetwork<float, ReLU<float>>.UnsafeLoad([
            [
                1f, 2f,
                3f, 4f,
            ],
            [
                5f, 6f,
                7f, 8f,
            ],
        ], [
            [0f, 0f],
            [0f, 0f],
        ]);

        Assert.Equal(2, network1.Inputs);
        Assert.Equal(2, network1.Outputs);
        Assert.Equal(3, network1.Layers.Length);
        Assert.Equal(2, network1.Layers[0].Inputs);
        Assert.Equal(2, network1.Layers[0].Size);
        Assert.Equal(2, network1.Layers[1].Inputs);
        Assert.Equal(2, network1.Layers[1].Size);
        Assert.Equal(2, network1.Layers[2].Inputs);
        Assert.Equal(2, network1.Layers[2].Size);

        Assert.Equal(2, network2.Inputs);
        Assert.Equal(2, network2.Outputs);
        Assert.Equal(2, network2.Layers.Length);
        Assert.Equal(2, network2.Layers[0].Inputs);
        Assert.Equal(2, network2.Layers[0].Size);
        Assert.Equal(2, network2.Layers[1].Inputs);
        Assert.Equal(2, network2.Layers[1].Size);
    }
}
