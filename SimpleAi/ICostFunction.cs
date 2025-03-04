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

public readonly struct MeanSquaredError<T> : ICostFunction<T>
    where T : INumberBase<T>,                             // T.CreateSaturating
    ISubtractionOperators<T, T, T>,                       // SubOp<T>
    IMultiplyOperators<T, T, T>,                          // Pow2Op<T>
    IAdditiveIdentity<T, T>, IAdditionOperators<T, T, T>, // AddOp<T>
    IDivisionOperators<T, T, T>                           // operator /
{
    /// <inheritdoc />
    public static T Calculate(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual)
        => // Sum(Pow(actual - expected, 2)) / N
            MathEx.Aggregate<T, BinaryUnaryPipeline<T, SubOp<T>, Pow2Op<T>>, AddOp<T>>(actual, expected)
            / T.CreateSaturating(actual.Length);

    /// <inheritdoc />
    public static void Derivative(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual, Span<T> outputs)
        => MathEx.Binary<T, SubOp<T>>(actual, expected, outputs);
}

public readonly struct CrossEntropy<T> : ICostFunction<T>
    where T : INumber<T>,             // T.Zero, T.One, T.Max, operator +, operator /
    IComparisonOperators<T, T, bool>, // T.Exp
    IExponentialFunctions<T>,         // operator >
    ILogarithmicFunctions<T>          // T.Log
{
    /// <inheritdoc />
    public static T Calculate(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual)
        => MathEx.Aggregate<T, CrossEntropyLoopOp, AddOp<T>>(expected, actual);

    /// <inheritdoc />
    public static void Derivative(ReadOnlySpan<T> expected, ReadOnlySpan<T> actual, Span<T> outputs)
        => MathEx.Binary<T, DerivativeOp>(expected, actual, outputs);

    private readonly struct CrossEntropyLoopOp : IBinOp<T>
    {
        public static T Execute(T label, T output)
        {
            T prediction = MathEx.Sigmoid(output);
            return label > T.Zero
                       ? -T.Log(T.Max(prediction, T.CreateSaturating(1e-8)))
                       : -T.Log(T.Max(T.One - prediction, T.CreateSaturating(1e-8)));
        }

        public static Vector<T> Execute(Vector<T> label, Vector<T> output)
        {
            if (typeof(T) == typeof(float))
            {
                Vector<T> prediction = MathEx.Sigmoid(output);
                return Vector.ConditionalSelect(
                                 Vector.GreaterThan(label.As<T, float>(), Vector<float>.Zero),
                                 -Vector.Log(Vector.Max(prediction.As<T, float>(), Vector.Create(1e-8f))),
                                 -Vector.Log(
                                     Vector.Max(Vector<float>.One - prediction.As<T, float>(), Vector.Create(1e-8f))))
                             .As<float, T>();
            }
            if (typeof(T) == typeof(double))
            {
                Vector<T> prediction = MathEx.Sigmoid(output);
                return Vector.ConditionalSelect(
                                 Vector.GreaterThan(label.As<T, double>(), Vector<double>.Zero),
                                 -Vector.Log(Vector.Max(prediction.As<T, double>(), Vector.Create(1e-8))),
                                 -Vector.Log(
                                     Vector.Max(Vector<double>.One - prediction.As<T, double>(), Vector.Create(1e-8))))
                             .As<double, T>();
            }
            else
            {
                Vector<T> res = Vector<T>.Zero;
                for (var idx = 0; idx < Vector<T>.Count; idx++)
                    res = res.WithElement(idx, Execute(label[idx], output[idx]));
                return res;
            }
        }
    }

    private readonly struct DerivativeOp : IBinOp<T>
    {
        /// <inheritdoc />
        public static T Execute(T label, T output)
        {
            T prediction = MathEx.Sigmoid(output);
            return label > T.Zero ? prediction - T.One : prediction;
        }

        /// <inheritdoc />
        public static Vector<T> Execute(Vector<T> labels, Vector<T> outputs)
        {
            Vector<T> predictions = MathEx.Sigmoid(outputs);
            return Vector.ConditionalSelect(
                Vector.GreaterThan(predictions, Vector<T>.Zero),
                predictions - Vector<T>.One,
                predictions);
        }
    }
}
