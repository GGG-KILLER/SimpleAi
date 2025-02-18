using System.Numerics;
using JetBrains.Annotations;
using SimpleAi.Math;

namespace SimpleAi;

/// <summary>
/// The interface for cost (also known as loss) functions.
/// </summary>
/// <typeparam name="T">The numeric type accepted by the cost function.</typeparam>
[PublicAPI]
public interface ICostFunction<T>
{
    /// <summary>
    /// Calculates the cost of the neural network based on the <paramref name="expected"/> outputs and the
    /// <paramref name="actual"/> inputs.
    /// </summary>
    /// <param name="expected">The inputs that were expected to be returned by the neural network.</param>
    /// <param name="actual">The inputs that were actually obtained when executing the inference.</param>
    /// <returns>The current cost of the neural network.</returns>
    [PublicAPI]
    static abstract T Calculate(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual);

    /// <summary>
    /// Calculates the derivative of the cost function at the provided <paramref name="expected"/> and <paramref name="actual"/>
    /// inputs and stores them on the <paramref name="outputs"/>.
    /// </summary>
    /// <param name="expected">The inputs that were expected to be returned by the neural network.</param>
    /// <param name="actual">The inputs that were actually obtained when executing the inference.</param>
    /// <param name="outputs">The results of the derivative calculation.</param>
    /// <returns></returns>
    [PublicAPI]
    static abstract void Derivative(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual, Span<T> outputs);
}

[PublicAPI]
public readonly struct NaiveSquaredError<T> : ICostFunction<T>
    where T : ISubtractionOperators<T, T, T>,             // SubOp<T>
    IMultiplyOperators<T, T, T>,                          // Pow2Op<T>
    IAdditiveIdentity<T, T>, IAdditionOperators<T, T, T>, // AddOp<T>, operator +
    INumberBase<T>                                        // T.One
{
    /// <inheritdoc />
    [PublicAPI]
    public static T Calculate(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual)
        =>
            // Pow(actual - expected, 2)
            MathEx.Aggregate<T, BinaryUnaryPipeline<T, SubOp<T>, Pow2Op<T>>, AddOp<T>>(actual, expected);

    /// <inheritdoc />
    [PublicAPI]
    public static void Derivative(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual, Span<T> outputs)
        => MathEx.Ternary<T, DoubleBinaryOp<T, SubOp<T>, MulOp<T>>>(expected, actual, T.One + T.One, outputs);
}

public readonly struct MeanSquaredError<T> : ICostFunction<T> where T : INumberBase<T> // T.One
{
    /// <inheritdoc />
    public static T Calculate(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual)
        => (T.One / (T.One + T.One))
           // Pow(actual - expected, 2)
           * MathEx.Aggregate<T, BinaryUnaryPipeline<T, SubOp<T>, Pow2Op<T>>, AddOp<T>>(actual, expected);

    /// <inheritdoc />
    public static void Derivative(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual, Span<T> outputs)
        => MathEx.Binary<T, SubOp<T>>(actual, expected, outputs);
}

public readonly struct CrossEntropy<T> : ICostFunction<T>
    where T : INumberBase<T> /* T.Zero, T.One */, ILogarithmicFunctions<T> /* T.Log */
{
    /// <inheritdoc />
    public static T Calculate(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual)
        => MathEx.Aggregate<T, CrossEntropyLoopOp, AddOp<T>>(expected, actual);

    /// <inheritdoc />
    public static void Derivative(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual, Span<T> outputs)
        => MathEx.Binary<T, DerivativeOp>(expected, actual, outputs);

    private readonly struct CrossEntropyLoopOp : IBinOp<T>
    {
        public static T Execute(T expected, T actual)
        {
            T res = expected == T.One ? -T.Log(actual) : -T.Log(T.One - actual);
            return T.IsNaN(res) ? T.Zero : res;
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

    private readonly struct DerivativeOp : IBinOp<T>
    {
        /// <inheritdoc />
        public static T Execute(T expected, T actual)
        {
            if (actual == T.Zero || actual == T.One) return T.Zero;
            return (-actual + expected) / (actual * (actual - T.One));
        }

        /// <inheritdoc />
        public static Vector<T> Execute(Vector<T> expected, Vector<T> actual)
        {
            var derivative = (-actual + expected) / (actual * (actual - Vector<T>.One));
            return Vector.ConditionalSelect(
                Vector.BitwiseOr(Vector.Equals(expected, Vector<T>.Zero), Vector.Equals(expected, Vector<T>.Zero)),
                Vector<T>.Zero,
                derivative);
        }
    }
}
