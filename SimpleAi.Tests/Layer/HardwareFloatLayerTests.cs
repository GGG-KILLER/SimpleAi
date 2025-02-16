using System.Reflection;
using JetBrains.Annotations;

namespace SimpleAi.Tests.Layer;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class HardwareFloatLayerTests
{
    public static TheoryData<Type> ExecutesCorrectlyOnFloatVectorizationPathData()
        => [.. ActivationHelper.GetActivationTypes<float>()];

    [Theory(
         Skip = "Vector<float> is not hardware accelerated.",
         SkipUnless = nameof(SkipConditions.FloatVectorsAreHardwareAccelerated),
         SkipType = typeof(SkipConditions)), MemberData(nameof(ExecutesCorrectlyOnFloatVectorizationPathData))]
    public void LayerX2ERunInference_Executes_correctly_on_float_vectorization_path(Type activationFunction)
    {
        typeof(HardwareFloatLayerTests).GetMethod(
                                           nameof(
                                               Layer_RunInference_Executes_correctly_on_float_vectorization_pathCore),
                                           BindingFlags.Static | BindingFlags.NonPublic)!
                                       .MakeGenericMethod(activationFunction)
                                       .Invoke(obj: null, parameters: null);
    }

    private static void Layer_RunInference_Executes_correctly_on_float_vectorization_pathCore<T>()
        where T : IActivationFunction<float>
    {
        Span<float> expected = stackalloc float[3];
        // @formatter:off
        Layer<float, T> layer = Layer<float, T>.LoadUnsafe([
             1f,  2f,  3f,  4f,  5f,  6f,  7f,  8f,  9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f, 2f,
            17f, 18f, 19f, 20f, 21f, 22f, 23f, 24f, 25f, 26f, 27f, 28f, 29f, 30f, 31f, 32f, 2f,
            33f, 34f, 35f, 36f, 37f, 38f, 39f, 40f, 41f, 42f, 43f, 44f, 45f, 46f, 47f, 48f, 2f,
        ], [
            1f,
            1f,
            1f,
        ]);
        T.Activate([
            (1f * 49f) + (2f * 50f) + (3f * 51f) + (4f * 52f) + (5f * 53f) + (6f * 54f) + (7f * 55f) + (8f * 56f) + (9f * 57f) + (10f * 58f) + (11f * 59f) + (12f * 60f) + (13f * 61f) + (14f * 62f) + (15f * 63f) + (16f * 64f) + (2f * 2f) + 1f,
            (17f * 49f) + (18f * 50f) + (19f * 51f) + (20f * 52f) + (21f * 53f) + (22f * 54f) + (23f * 55f) + (24f * 56f) + (25f * 57f) + (26f * 58f) + (27f * 59f) + (28f * 60f) + (29f * 61f) + (30f * 62f) + (31f * 63f) + (32f * 64f) + (2f * 2f) + 1f,
            (33f * 49f) + (34f * 50f) + (35f * 51f) + (36f * 52f) + (37f * 53f) + (38f * 54f) + (39f * 55f) + (40f * 56f) + (41f * 57f) + (42f * 58f) + (43f * 59f) + (44f * 60f) + (45f * 61f) + (46f * 62f) + (47f * 63f) + (48f * 64f) + (2f * 2f) + 1f,
        ], expected);
        // @formatter:on

        Span<float> output = stackalloc float[3];
        layer.RunInference(
            [49f, 50f, 51f, 52f, 53f, 54f, 55f, 56f, 57f, 58f, 59f, 60f, 61f, 62f, 63f, 64f, 2f],
            output);

        Assert.Equal(expected, output);
    }
}
