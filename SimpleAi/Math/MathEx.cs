using System.Numerics;

namespace SimpleAi.Math;

internal static partial class MathEx
{
    public static int DivideRoundingUp(int dividend, int divisor) => (dividend + divisor - 1) / divisor;

    public static T RandomBetweenNormalDistribution<T>(Random random, T mean, T stdDev) where T : INumberBase<T>
    {
        double x1 = 1 - random.NextDouble();
        double x2 = 1 - random.NextDouble();

        double y1 = double.Sqrt(-2.0 * double.Log(x1)) * double.Cos(2.0 * double.Pi * x2);
        return T.CreateSaturating((y1 * double.CreateSaturating(stdDev)) + double.CreateSaturating(mean));
    }
}
