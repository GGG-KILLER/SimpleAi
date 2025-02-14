namespace SimpleAi.Tests.NeuralNetwork;

public class NeuralNetworkLoadingTests
{
    [Fact]
    public void NeuralNetworkX2EUnsafeLoad_Correctly_sets_inputs_and_outputs_as_well_as_loads_layers()
    {
        // @formatter:off
        NeuralNetwork<float, ReLU<float>> network1 = NeuralNetwork<float, ReLU<float>>.UnsafeLoad([
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

        NeuralNetwork<float, ReLU<float>> network2 = NeuralNetwork<float, ReLU<float>>.UnsafeLoad([
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
        // @formatter:on

        Assert.Equal(expected: 2, network1.Inputs);
        Assert.Equal(expected: 2, network1.Outputs);
        Assert.Equal(expected: 3, network1.Layers.Length);
        Assert.Equal(expected: 2, network1.Layers[index: 0].Inputs);
        Assert.Equal(expected: 2, network1.Layers[index: 0].Size);
        Assert.Equal(expected: 2, network1.Layers[index: 1].Inputs);
        Assert.Equal(expected: 2, network1.Layers[index: 1].Size);
        Assert.Equal(expected: 2, network1.Layers[index: 2].Inputs);
        Assert.Equal(expected: 2, network1.Layers[index: 2].Size);

        Assert.Equal(expected: 2, network2.Inputs);
        Assert.Equal(expected: 2, network2.Outputs);
        Assert.Equal(expected: 2, network2.Layers.Length);
        Assert.Equal(expected: 2, network2.Layers[index: 0].Inputs);
        Assert.Equal(expected: 2, network2.Layers[index: 0].Size);
        Assert.Equal(expected: 2, network2.Layers[index: 1].Inputs);
        Assert.Equal(expected: 2, network2.Layers[index: 1].Size);
    }
}
