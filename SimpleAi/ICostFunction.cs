using System.Numerics;
using System.Runtime.CompilerServices;
using SimpleAi.Math;

namespace SimpleAi;

public interface ICostFunction<T>
{
    static abstract T Calculate(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual);
}

public readonly struct NaiveSquaredError<T> : ICostFunction<T>
    where T : ISubtractionOperators<T, T, T>, IMultiplyOperators<T, T, T>, IAdditiveIdentity<T, T>,
    IAdditionOperators<T, T, T>
{
    public static T Calculate(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual)
        => MathEx.Aggregate<T, BinaryUnaryPipeline<T, SubOp<T>, Pow2Op<T>>, AddOp<T>>(expected, actual);
}

public readonly struct MeanSquaredError<T> : ICostFunction<T> where T : INumberBase<T> // T.One
{
    public static T Calculate(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual)
        => (T.One / (T.One + T.One))
           * MathEx.Aggregate<T, BinaryUnaryPipeline<T, SubOp<T>, Pow2Op<T>>, AddOp<T>>(expected, actual);
}

public readonly struct CrossEntropy<T> : ICostFunction<T>
    where T : INumberBase<T> /* T.Zero, T.One */, ILogarithmicFunctions<T> /* T.Log */
{
    public static T Calculate(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual)
        => MathEx.Aggregate<T, CrossEntropyLoopOp, AddOp<T>>(expected, actual);

    private readonly struct CrossEntropyLoopOp : IBinOp<T>
    {
        public static T Execute(T expected, T actual)
        {
            T res = expected == T.One ? -T.Log(actual) : -T.Log(T.One - actual);
            if (typeof(T) == typeof(Half))
                res = Half.IsNaN(Unsafe.BitCast<T, Half>(res)) ? T.Zero : res;
            else if (typeof(T) == typeof(float))
                res                                   = float.IsNaN(Unsafe.BitCast<T, float>(res)) ? T.Zero : res;
            else if (typeof(T) == typeof(double)) res = double.IsNaN(Unsafe.BitCast<T, double>(res)) ? T.Zero : res;
            return res;
        }

        public static Vector<T> Execute(Vector<T> expected, Vector<T> actual)
        {
            if (typeof(T) == typeof(float))
            {
                Vector<float> nlog   = -Vector.Log(actual.As<T, float>());
                Vector<float> nlogm1 = -Vector.Log(Vector<float>.One - actual.As<T, float>());
                Vector<float> res = Vector.ConditionalSelect(
                    Vector.Equals(expected.As<T, float>(), Vector<float>.One),
                    nlog,
                    nlogm1);
                return Vector.ConditionalSelect(Vector.IsNaN(res), Vector<float>.Zero, res).As<float, T>();
            }
            if (typeof(T) == typeof(double))
            {
                Vector<double> nlog   = -Vector.Log(actual.As<T, double>());
                Vector<double> nlogm1 = -Vector.Log(Vector<double>.One - actual.As<T, double>());
                Vector<double> res = Vector.ConditionalSelect(
                    Vector.Equals(expected.As<T, double>(), Vector<double>.One),
                    nlog,
                    nlogm1);
                return Vector.ConditionalSelect(Vector.IsNaN(res), Vector<double>.Zero, res).As<double, T>();
            }
            else
            {
                Vector<T> res = Vector<T>.Zero;
                for (var idx = 0; idx < Vector<T>.Count; idx++)
                    res = res.WithElement(idx, Execute(expected[idx], actual[idx]));
                return res;
            }
        }
    }
}
