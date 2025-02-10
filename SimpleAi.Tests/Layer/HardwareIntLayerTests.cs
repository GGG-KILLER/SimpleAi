using System.Reflection;

namespace SimpleAi.Tests.Layer;

public class HardwareIntLayerTests
{
    public static TheoryData<Type> ExecutesCorrectlyOnIntegerVectorizationPathData() => [.. ActivationHelper.GetNonExponentiatingActivationTypes<int>()];
    [Theory(Skip = "Vector<int> is not hardware accelerated.", SkipUnless = nameof(SkipConditions.IntVectorsAreHardwareAccelerated), SkipType = typeof(SkipConditions))]
    [MemberData(nameof(ExecutesCorrectlyOnIntegerVectorizationPathData))]
    public void Layer_RunInference_ExecutesCorrectlyOnIntegerVectorizationPath(Type activationFunction)
    {
        typeof(HardwareIntLayerTests).GetMethod(nameof(Layer_RunInference_ExecutesCorrectlyOnIntegerVectorizationPathCore), BindingFlags.Static | BindingFlags.NonPublic)!
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
}
