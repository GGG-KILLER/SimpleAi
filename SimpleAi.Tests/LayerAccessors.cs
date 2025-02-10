using System.Numerics;
using System.Reflection;

namespace SimpleAi.Tests;

public static class LayerAccessors
{
    private const BindingFlags F = BindingFlags.NonPublic | BindingFlags.Instance;

    public static T[] GetWeights<T>(Layer<T, ReLU<T>> layer) where T : INumber<T> =>
    (T[])layer.GetType().GetField("_weights", F)!.GetValue(layer)!;

    public static T[] GetBiases<T>(Layer<T, ReLU<T>> layer) where T : INumber<T> =>
        (T[])layer.GetType().GetField("_biases", F)!.GetValue(layer)!;
}
