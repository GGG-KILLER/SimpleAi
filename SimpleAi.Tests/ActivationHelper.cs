using System.Numerics;
using System.Reflection;

namespace SimpleAi.Tests;

internal static class ActivationHelper
{
    private static readonly MethodInfo s_activateMethodInfo = typeof(ActivationHelper).GetMethod(
        nameof(Activate),
        genericParameterCount: 2,
        BindingFlags.Static | BindingFlags.NonPublic,
        [typeof(int)])!;

    public static IEnumerable<Type> GetNonExponentiatingActivationTypes<T>() where T : INumber<T> => [typeof(ReLU<T>)];

    public static IEnumerable<Type> GetActivationTypes<T>() where T : INumber<T>, IExponentialFunctions<T>
        => [typeof(Sigmoid<T>), typeof(TanH<T>), typeof(ReLU<T>), typeof(SoftMax<T>)];

    public static T Activate<T, TActivation>(T value)
        where T : unmanaged, INumber<T> where TActivation : IActivationFunction<T>
    {
        Span<T> output = stackalloc T[1];
        TActivation.Activate([value], output);
        return output[index: 0];
    }

    public static T Activate<T>(Type activationFunction, T value)
        => (T) s_activateMethodInfo.MakeGenericMethod(typeof(T), activationFunction).Invoke(obj: null, [value])!;
}
