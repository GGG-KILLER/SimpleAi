using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi.Tests.Layer;

[UsedImplicitly(ImplicitUseTargetFlags.WithMembers)]
public class SoftwareLayerTests
{
    [Fact]
    public void LayerX2ERunInference_Executes_correctly_on_software_fallback()
    {
        // @formatter:off
        Layer<float, ReLu<float>> layer = new Layer<float, ReLu<float>>(Tensor.Create([
            1f, 2f,
            3f, 4f,
        ], [2, 2]), (float[])[
            1f,
            1f,
        ]);
        var expected = ReLu<float>.Activate((float[])[
            (1f * 7f) + (2f * 9f) + 1f,
            (3f * 7f) + (4f * 9f) + 1f,
        ]);
        // @formatter:on

        var output = layer.RunInference((float[]) [7f, 9f]);

        Assert.Equal(expected, output);
    }
}
