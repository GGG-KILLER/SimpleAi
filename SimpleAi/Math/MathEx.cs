using System.Numerics;

namespace SimpleAi.Math;

internal static partial class MathEx
{
    public static int DivideRoundingUp(int dividend, int divisor) => (dividend + divisor - 1) / divisor;

    public static T Sigmoid<T>(T value)
        where T : INumberBase<T>,         // T.Zero, T.One, operator +, operator /
        IComparisonOperators<T, T, bool>, // T.Exp
        IExponentialFunctions<T>          // operator >
    {
        if (value > T.Zero) return T.One / (T.One + T.Exp(value));

        var exp = T.Exp(value);
        return exp / (T.One + exp);
    }

    public static Vector<T> Sigmoid<T>(Vector<T> value)
        where T : INumberBase<T>,         // T.Zero, T.One, operator +, operator /
        IComparisonOperators<T, T, bool>, // T.Exp
        IExponentialFunctions<T>          // operator >
    {
        if (typeof(T) == typeof(float))
        {
            var exp   = Vector.Exp(value.As<T, float>());
            var expP1 = Vector<float>.One + exp;
            return Vector.ConditionalSelect(
                Vector.GreaterThan(value.As<T, float>(), Vector<float>.Zero),
                Vector<float>.One / expP1,
                exp / expP1).As<float, T>();
        }
        else if (typeof(T) == typeof(double))
        {
            var exp   = Vector.Exp(value.As<T, double>());
            var expP1 = Vector<double>.One + exp;
            return Vector.ConditionalSelect(
                Vector.GreaterThan(value.As<T, double>(), Vector<double>.Zero),
                Vector<double>.One / expP1,
                exp / expP1).As<double, T>();
        }
        else
        {
            Vector<T> res = Vector<T>.Zero;

            for (var idx = 0; idx < Vector<T>.Count; idx++) res = res.WithElement(idx, Sigmoid(value[idx]));

            return res;
        }
    }

    public static T RandomBetweenNormalDistribution<T>(Random random, T mean, T stdDev) where T : INumberBase<T>
    {
        double x1 = 1 - random.NextDouble();
        double x2 = 1 - random.NextDouble();

        double y1 = double.Sqrt(-2.0 * double.Log(x1)) * double.Cos(2.0 * double.Pi * x2);
        return T.CreateSaturating((y1 * double.CreateSaturating(stdDev)) + double.CreateSaturating(mean));
    }
}
