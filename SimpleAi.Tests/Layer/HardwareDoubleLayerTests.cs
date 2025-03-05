using System.Numerics.Tensors;
using System.Reflection;
using JetBrains.Annotations;

namespace SimpleAi.Tests.Layer;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class HardwareDoubleLayerTests
{
    public static TheoryData<Type> ExecutesCorrectlyOnDoubleVectorizationPathData()
        => [.. ActivationHelper.GetActivationTypes<double>()];

    [Theory(
         Skip = "Vector<double> is not hardware accelerated.",
         SkipUnless = nameof(SkipConditions.DoubleVectorsAreHardwareAccelerated),
         SkipType = typeof(SkipConditions)), MemberData(nameof(ExecutesCorrectlyOnDoubleVectorizationPathData))]
    public void LayerX2ERunInference_Executes_correctly_on_double_vectorization_path(Type activationFunction)
    {
        typeof(HardwareDoubleLayerTests).GetMethod(
                                            nameof(
                                                Layer_RunInference_Executes_correctly_on_double_vectorization_pathCore),
                                            BindingFlags.Static | BindingFlags.NonPublic)!
                                        .MakeGenericMethod(activationFunction)
                                        .Invoke(obj: null, parameters: null);
    }

    private static void Layer_RunInference_Executes_correctly_on_double_vectorization_pathCore<T>()
        where T : IActivationFunction<double>
    {
        // @formatter:off
        Layer<double, T> layer = new Layer<double, T>(Tensor.Create([
            1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 2.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 2.0,
            33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 2.0,
        ], [3, 17]), Tensor.Create([
            1.0,
            1.0,
            1.0,
        ], [3]));
        var expected = T.Activate((double[])[
            (1.0 * 49.0) + (2.0 * 50.0) + (3.0 * 51.0) + (4.0 * 52.0) + (5.0 * 53.0) + (6.0 * 54.0) + (7.0 * 55.0) + (8.0 * 56.0) + (9.0 * 57.0) + (10.0 * 58.0) + (11.0 * 59.0) + (12.0 * 60.0) + (13.0 * 61.0) + (14.0 * 62.0) + (15.0 * 63.0) + (16.0 * 64.0) + (2.0 * 2.0) + 1.0,
            (17.0 * 49.0) + (18.0 * 50.0) + (19.0 * 51.0) + (20.0 * 52.0) + (21.0 * 53.0) + (22.0 * 54.0) + (23.0 * 55.0) + (24.0 * 56.0) + (25.0 * 57.0) + (26.0 * 58.0) + (27.0 * 59.0) + (28.0 * 60.0) + (29.0 * 61.0) + (30.0 * 62.0) + (31.0 * 63.0) + (32.0 * 64.0) + (2.0 * 2.0) + 1.0,
            (33.0 * 49.0) + (34.0 * 50.0) + (35.0 * 51.0) + (36.0 * 52.0) + (37.0 * 53.0) + (38.0 * 54.0) + (39.0 * 55.0) + (40.0 * 56.0) + (41.0 * 57.0) + (42.0 * 58.0) + (43.0 * 59.0) + (44.0 * 60.0) + (45.0 * 61.0) + (46.0 * 62.0) + (47.0 * 63.0) + (48.0 * 64.0) + (2.0 * 2.0) + 1.0,
        ]);
        // @formatter:on

        var output = layer.RunInference(
            (double[])
            [
                49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 2.0,
            ]);

        Assert.Equal(expected[0], output[0]);
        Assert.Equal(expected[1], output[1]);
        Assert.Equal(expected[2], output[2]);
    }
}
