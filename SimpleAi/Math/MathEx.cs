using System.Numerics;
using System.Runtime.CompilerServices;

namespace SimpleAi.Math;

internal static partial class MathEx
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int DivideRoundingUp(int number, int divisor) => (number + divisor - 1) / divisor;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T ReLU<T>(T value) where T : INumber<T> =>
        T.Max(T.Zero, value);
}
