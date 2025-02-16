using JetBrains.Annotations;

namespace SimpleAi.Tests.Layer;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class SoftwareLayerTests
{
    [Fact]
    public void LayerX2ERunInference_Executes_correctly_on_software_fallback()
    {
        Span<int> expected = stackalloc int[3];
        // @formatter:off
        Layer<int, ReLU<int>> layer = Layer<int, ReLU<int>>.LoadUnsafe([
            1, 2,
            3, 4,
            5, 6,
        ], [
            1,
            1,
            1,
        ]);
        ReLU<int>.Activate([
            1 * 7 + 2 * 9 + 1,
            3 * 7 + 4 * 9 + 1,
            5 * 7 + 6 * 9 + 1,
        ], expected);
        // @formatter:on

        Span<int> output = stackalloc int[3];
        layer.RunInference([7, 9], output);

        Assert.Equal(expected, output);
    }
}
