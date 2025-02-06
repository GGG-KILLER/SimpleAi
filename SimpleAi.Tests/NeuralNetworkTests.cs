namespace SimpleAi.Tests;

public class NeuralNetworkTests
{
    [Fact]
    public void NeuralNetwork_Constructor_ThrowsExceptionOnNegativeOrZeroInput()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuralNetwork<float>(-1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NeuralNetwork<float>(0, 1));
    }

    [Fact]
    public void NeuralNetwork_Constructor_ThrowsExceptionWithoutAnyLayerSizes()
    {
        Assert.Throws<ArgumentException>(() => new NeuralNetwork<float>(1));
    }

    [Fact]
    public void NeuralNetwork_Constructor_ThrowsExceptionWithNegativeOrZeroedLayerSizes()
    {
        Assert.Throws<ArgumentException>(() => new NeuralNetwork<float>(1, -1));
        Assert.Throws<ArgumentException>(() => new NeuralNetwork<float>(1, 0));
    }

    [Fact]
    public void NeuralNetwork_Constructor_CreatesCorrectLayerStructure()
    {
        var network = new NeuralNetwork<double>(2, 5, 5, 3);


        Assert.Equal(2, network.Inputs);
        Assert.Equal(3, network.Outputs);

        Assert.Equal(3, network.Layers.Length);
        Assert.Equal(2, network.Layers[0].Inputs);
        Assert.Equal(5, network.Layers[0].Size);
        Assert.Equal(5, network.Layers[1].Inputs);
        Assert.Equal(5, network.Layers[1].Size);
        Assert.Equal(5, network.Layers[2].Inputs);
        Assert.Equal(3, network.Layers[2].Size);
    }

    [Fact]
    public void NeuralNetwork_UnsafeLoad_CorrectlyLoadsNetworkComponents()
    {
        var network1 = NeuralNetwork<float>.UnsafeLoad([
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

        var network2 = NeuralNetwork<float>.UnsafeLoad([
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

    [Fact]
    public void NeuralNetwork_Randomize_ProperlyRandomizesLayers()
    {
        var network = new NeuralNetwork<long>(2, 3);

        network.Randomize(5);
        network.Randomize(10);
        network.Randomize(15);

        foreach (var layer in network.Layers)
        {
            var weights = LayerAccessors.GetWeights(layer);
            var biases = LayerAccessors.GetBiases(layer);

            Assert.Contains(weights, x => x != 0);
            Assert.Contains(biases, x => x != 0);
        }
    }

    [Fact]
    public void NeuralNetwork_RunInference_ThrowsExceptionOnWrongInputSize()
    {
        var network = new NeuralNetwork<float>(2, 2);

        Assert.Throws<ArgumentException>(() =>
            network.RunInference([], stackalloc float[2]));
        Assert.Throws<ArgumentException>(() =>
            network.RunInference([1], stackalloc float[2]));
        Assert.Throws<ArgumentException>(() =>
            network.RunInference([1, 1, 1], stackalloc float[2]));
    }

    [Fact]
    public void NeuralNetwork_RunInference_ThrowsExceptionOnWrongOutputSize()
    {
        var network = new NeuralNetwork<float>(2, 2);

        Assert.Throws<ArgumentException>(() =>
            network.RunInference([1, 1], []));
        Assert.Throws<ArgumentException>(() =>
            network.RunInference([1, 1], [1]));
        Assert.Throws<ArgumentException>(() =>
            network.RunInference([1, 1], [1, 1, 1]));
    }

    [Fact]
    public void NeuralNetwork_RunInference_RunsProperlyWithOddNumberOfLayers()
    {
        var network = NeuralNetwork<float>.UnsafeLoad([
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

        Span<float> output = stackalloc float[2];
        network.RunInference([13, 14], output);

        // First hidden layer
        const float a11 = 1f * 13f + 2f * 14f + 0f;
        const float a12 = 3f * 13f + 4f * 14f + 0f;

        // Second hidden layer
        const float a21 = 5f * a11 + 6f * a12 + 0f;
        const float a22 = 7f * a11 + 8f * a12 + 0f;

        // Output Layer
        const float a31 = 9f * a21 + 10f * a22 + 0f;
        const float a32 = 11f * a21 + 12f * a22 + 0f;

        Assert.Equal(a31, output[0]);
        Assert.Equal(a32, output[1]);
    }

    [Fact]
    public void NeuralNetwork_RunInference_RunsProperlyWithEvenNumberOfLayers()
    {
        var network = NeuralNetwork<float>.UnsafeLoad([
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

        Span<float> output = stackalloc float[2];
        network.RunInference([13, 14], output);

        // First hidden layer
        const float a11 = 1f * 13f + 2f * 14f + 0f;
        const float a12 = 3f * 13f + 4f * 14f + 0f;

        // Output Layer
        const float a21 = 5f * a11 + 6f * a12 + 0f;
        const float a22 = 7f * a11 + 8f * a12 + 0f;

        Assert.Equal(a21, output[0]);
        Assert.Equal(a22, output[1]);
    }
}
