using System.Numerics;
using System.Reflection;

namespace SimpleAi.Tests;

public class LayerTests
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

    [Fact]
    public void Layer_LoadUnsafe_CorrectlyCopiesValuesIntoFields()
    {
        var layer = Layer<ulong, ReLU<ulong>>.LoadUnsafe([1UL, 2UL, 3UL, 4UL, 5UL, 6UL], [7UL, 8UL]);

        Assert.Equal(3, layer.Inputs);
        Assert.Equal(2, layer.Size);

        Assert.Equal([1UL, 2UL, 3UL, 4UL, 5UL, 6UL], LayerAccessors.GetWeights(layer));
        Assert.Equal([7Ul, 8UL], LayerAccessors.GetBiases(layer));
    }

    [Fact]
    public void Layer_RunInference_ExecutesCorrectlyOnSoftwareFallback()
    {
        var layer = Layer<int, ReLU<int>>.LoadUnsafe([
            1, 2,
            3, 4,
            5, 6,
        ], [
            1,
            1,
            1,
        ]);
        Span<int> expected = stackalloc int[3];
        ReLU<int>.Activate([
            1 * 7 + 2 * 9 + 1,
            3 * 7 + 4 * 9 + 1,
            5 * 7 + 6 * 9 + 1,
        ], expected);

        Span<int> output = stackalloc int[3];
        layer.RunInference([7, 9], output);

        Assert.Equal(expected, output);
    }

    public static TheoryData<Type> ExecutesCorrectlyOnDoubleVectorizationPathData() => [.. ActivationHelper.GetNonExponentiatingActivationTypes<double>()];
    [Theory]
    [MemberData(nameof(ExecutesCorrectlyOnDoubleVectorizationPathData))]
    public void Layer_RunInference_ExecutesCorrectlyOnDoubleVectorizationPath(Type activationFunction)
    {
        typeof(LayerTests).GetMethod(nameof(Layer_RunInference_ExecutesCorrectlyOnDoubleVectorizationPathCore), BindingFlags.Static | BindingFlags.NonPublic)!
            .MakeGenericMethod(activationFunction)
            .Invoke(null, null);
    }
    private static void Layer_RunInference_ExecutesCorrectlyOnDoubleVectorizationPathCore<T>() where T : IActivationFunction<double>
    {
        var layer = Layer<double, T>.LoadUnsafe([
            1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 2.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 2.0,
            33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 2.0,
        ], [
                1.0,
            1.0,
            1.0,
        ]);
        Span<double> expected = stackalloc double[3];
        T.Activate([
            1.0 * 49.0 + 2.0 * 50.0 + 3.0 * 51.0 + 4.0 * 52.0 + 5.0 * 53.0 + 6.0 * 54.0 + 7.0 * 55.0 + 8.0 * 56.0 + 9.0 * 57.0 + 10.0 * 58.0 + 11.0 * 59.0 + 12.0 * 60.0 + 13.0 * 61.0 + 14.0 * 62.0 + 15.0 * 63.0 + 16.0 * 64.0 + 2.0 * 2.0 + 1.0,
            17.0 * 49.0 + 18.0 * 50.0 + 19.0 * 51.0 + 20.0 * 52.0 + 21.0 * 53.0 + 22.0 * 54.0 + 23.0 * 55.0 + 24.0 * 56.0 + 25.0 * 57.0 + 26.0 * 58.0 + 27.0 * 59.0 + 28.0 * 60.0 + 29.0 * 61.0 + 30.0 * 62.0 + 31.0 * 63.0 + 32.0 * 64.0 + 2.0 * 2.0 + 1.0,
            33.0 * 49.0 + 34.0 * 50.0 + 35.0 * 51.0 + 36.0 * 52.0 + 37.0 * 53.0 + 38.0 * 54.0 + 39.0 * 55.0 + 40.0 * 56.0 + 41.0 * 57.0 + 42.0 * 58.0 + 43.0 * 59.0 + 44.0 * 60.0 + 45.0 * 61.0 + 46.0 * 62.0 + 47.0 * 63.0 + 48.0 * 64.0 + 2.0 * 2.0 + 1.0,
        ], expected);

        Span<double> output = stackalloc double[3];
        layer.RunInference([
                49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 2.0,
        ], output);

        Assert.Equal(expected[0], output[0]);
        Assert.Equal(expected[1], output[1]);
        Assert.Equal(expected[2], output[2]);
    }

    public static TheoryData<Type> ExecutesCorrectlyOnFloatVectorizationPathData() => [.. ActivationHelper.GetNonExponentiatingActivationTypes<float>()];
    [Theory]
    [MemberData(nameof(ExecutesCorrectlyOnFloatVectorizationPathData))]
    public void Layer_RunInference_ExecutesCorrectlyOnFloatVectorizationPath(Type activationFunction)
    {
        typeof(LayerTests).GetMethod(nameof(Layer_RunInference_ExecutesCorrectlyOnFloatVectorizationPathCore), BindingFlags.Static | BindingFlags.NonPublic)!
            .MakeGenericMethod(activationFunction)
            .Invoke(null, null);
    }
    private static void Layer_RunInference_ExecutesCorrectlyOnFloatVectorizationPathCore<T>()
        where T : IActivationFunction<float>
    {
        var layer = Layer<float, T>.LoadUnsafe([
             1f,  2f,  3f,  4f,  5f,  6f,  7f,  8f,  9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f, 2f,
            17f, 18f, 19f, 20f, 21f, 22f, 23f, 24f, 25f, 26f, 27f, 28f, 29f, 30f, 31f, 32f, 2f,
            33f, 34f, 35f, 36f, 37f, 38f, 39f, 40f, 41f, 42f, 43f, 44f, 45f, 46f, 47f, 48f, 2f,
        ], [
            1f,
            1f,
            1f,
        ]);
        Span<float> expected = stackalloc float[3];
        T.Activate([
            1f * 49f + 2f * 50f + 3f * 51f + 4f * 52f + 5f * 53f + 6f * 54f + 7f * 55f + 8f * 56f + 9f * 57f + 10f * 58f + 11f * 59f + 12f * 60f + 13f * 61f + 14f * 62f + 15f * 63f + 16f * 64f + 2f * 2f + 1f,
            17f * 49f + 18f * 50f + 19f * 51f + 20f * 52f + 21f * 53f + 22f * 54f + 23f * 55f + 24f * 56f + 25f * 57f + 26f * 58f + 27f * 59f + 28f * 60f + 29f * 61f + 30f * 62f + 31f * 63f + 32f * 64f + 2f * 2f + 1f,
            33f * 49f + 34f * 50f + 35f * 51f + 36f * 52f + 37f * 53f + 38f * 54f + 39f * 55f + 40f * 56f + 41f * 57f + 42f * 58f + 43f * 59f + 44f * 60f + 45f * 61f + 46f * 62f + 47f * 63f + 48f * 64f + 2f * 2f + 1f,
        ], expected);

        Span<float> output = stackalloc float[3];
        layer.RunInference([
            49f, 50f, 51f, 52f, 53f, 54f, 55f, 56f, 57f, 58f, 59f, 60f, 61f, 62f, 63f, 64f, 2f,
        ], output);

        Assert.Equal(expected, output);
    }

    public static TheoryData<Type> ExecutesCorrectlyOnIntegerVectorizationPathData() => [typeof(ReLU<int>)];
    [Theory]
    [MemberData(nameof(ExecutesCorrectlyOnIntegerVectorizationPathData))]
    public void Layer_RunInference_ExecutesCorrectlyOnIntegerVectorizationPath(Type activationFunction)
    {
        typeof(LayerTests).GetMethod(nameof(Layer_RunInference_ExecutesCorrectlyOnIntegerVectorizationPathCore), BindingFlags.Static | BindingFlags.NonPublic)!
            .MakeGenericMethod(activationFunction)
            .Invoke(null, null);
    }
    private static void Layer_RunInference_ExecutesCorrectlyOnIntegerVectorizationPathCore<T>()
        where T : IActivationFunction<int>
    {
        var layer = Layer<int, T>.LoadUnsafe([
             1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 2,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 2,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 2,
        ], [
            1,
            1,
            1,
        ]);
        Span<int> expected = stackalloc int[3];
        T.Activate([
            1 * 49 + 2 * 50 + 3 * 51 + 4 * 52 + 5 * 53 + 6 * 54 + 7 * 55 + 8 * 56 + 9 * 57 + 10 * 58 + 11 * 59 + 12 * 60 + 13 * 61 + 14 * 62 + 15 * 63 + 16 * 64 + 2 * 2 + 1,
            17 * 49 + 18 * 50 + 19 * 51 + 20 * 52 + 21 * 53 + 22 * 54 + 23 * 55 + 24 * 56 + 25 * 57 + 26 * 58 + 27 * 59 + 28 * 60 + 29 * 61 + 30 * 62 + 31 * 63 + 32 * 64 + 2 * 2 + 1,
            33 * 49 + 34 * 50 + 35 * 51 + 36 * 52 + 37 * 53 + 38 * 54 + 39 * 55 + 40 * 56 + 41 * 57 + 42 * 58 + 43 * 59 + 44 * 60 + 45 * 61 + 46 * 62 + 47 * 63 + 48 * 64 + 2 * 2 + 1,
        ], expected);

        Span<int> output = stackalloc int[3];
        layer.RunInference([
            49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 2,
        ], output);

        Assert.Equal(expected, output);
    }


    private static TheoryData<Type> SingleArgumentActivationFunctionTypeTestData<T>()
        where T : INumber<T>, IExponentialFunctions<T>
    {
        return [
            typeof(Sigmoid<T>),
            typeof(TanH<T>),
            typeof(ReLU<T>),
            typeof(SoftMax<T>)
        ];
    }
}

public static class LayerAccessors
{
    private const BindingFlags F = BindingFlags.NonPublic | BindingFlags.Instance;

    public static T[] GetWeights<T>(Layer<T, ReLU<T>> layer) where T : INumber<T> =>
    (T[])layer.GetType().GetField("_weights", F)!.GetValue(layer)!;

    public static T[] GetBiases<T>(Layer<T, ReLU<T>> layer) where T : INumber<T> =>
        (T[])layer.GetType().GetField("_biases", F)!.GetValue(layer)!;
}
