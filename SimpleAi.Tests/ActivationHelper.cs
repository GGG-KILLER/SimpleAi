using System.Numerics;

namespace SimpleAi.Tests;

internal static class ActivationHelper
{
    public static IEnumerable<Type> GetActivationTypes<T>() where T : INumber<T>, IExponentialFunctions<T>
        => [typeof(Sigmoid<T>), typeof(TanH<T>), typeof(ReLu<T>), typeof(SiLu<T>), typeof(SoftMax<T>)];
}
