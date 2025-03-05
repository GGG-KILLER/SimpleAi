using System.Diagnostics;
using System.Numerics;
using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi;

/// <summary>The interface for cost (also known as loss) functions.</summary>
/// <typeparam name="T">The numeric type accepted by the cost function.</typeparam>
[PublicAPI]
public interface ICostFunction<T>
{
    /// <summary>
    ///     Calculates the cost of the neural network based on the <paramref name="expected" /> outputs and the
    ///     <paramref name="actual" /> inputs.
    /// </summary>
    /// <param name="expected">The inputs that were expected to be returned by the neural network.</param>
    /// <param name="actual">The inputs that were actually obtained when executing the inference.</param>
    /// <returns>The current cost of the neural network.</returns>
    [PublicAPI]
    static abstract T Calculate(in ReadOnlyTensorSpan<T> expected, in ReadOnlyTensorSpan<T> actual);

    /// <summary>
    ///     Calculates the derivative of the cost function at the provided <paramref name="expected" /> and
    ///     <paramref name="actual" /> inputs.
    /// </summary>
    /// <param name="expected">The inputs that were expected to be returned by the neural network.</param>
    /// <param name="actual">The inputs that were actually obtained when executing the inference.</param>
    /// <returns>The results of executing the derivative.</returns>
    [PublicAPI]
    static abstract Tensor<T> Derivative(in ReadOnlyTensorSpan<T> expected, in ReadOnlyTensorSpan<T> actual);
}

public readonly struct MeanSquaredError<T> : ICostFunction<T>
    where T : ISubtractionOperators<T, T, T>,                   // Tensor.Subtract
    IMultiplyOperators<T, T, T>, IMultiplicativeIdentity<T, T>, // Tensor.Multiply
    IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>,       // Tensor.Sum
    INumber<T>                                                  // T.CreateSaturating
{
    /// <inheritdoc />
    public static T Calculate(in ReadOnlyTensorSpan<T> expected, in ReadOnlyTensorSpan<T> actual)
    {
        if (!expected.Lengths.SequenceEqual(actual.Lengths))
            throw new ArgumentException(message: "Lengths of both input tensors don't match.");

        Tensor<T> error = Tensor.Subtract(actual, expected);
        return Tensor.Sum<T>(Tensor.Multiply<T>(error, error)) / T.CreateSaturating(actual.FlattenedLength);
    }

    /// <inheritdoc />
    public static Tensor<T> Derivative(in ReadOnlyTensorSpan<T> expected, in ReadOnlyTensorSpan<T> actual)
    {
        if (!expected.Lengths.SequenceEqual(actual.Lengths))
            throw new ArgumentException(message: "Lengths of both input tensors don't match.");
        return Tensor.Subtract(actual, expected);
    }
}

public readonly struct CrossEntropy<T> : ICostFunction<T>
    where T : IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>, // T.AdditiveIdentity, operator +
    ILogarithmicFunctions<T>,                                       // T.Log
    INumberBase<T>,                                                 // T.One, T.Zero, T.IsNaN
    IEqualityOperators<T, T, bool>                                  // operator ==
{
    /// <inheritdoc />
    public static T Calculate(in ReadOnlyTensorSpan<T> expected, in ReadOnlyTensorSpan<T> actual)
    {
        if (!expected.Lengths.SequenceEqual(actual.Lengths))
            throw new ArgumentException(message: "Lengths of both input tensors don't match.");

        var cost               = T.AdditiveIdentity;
        var expectedEnumerator = expected.GetEnumerator();
        var actualEnumerator   = actual.GetEnumerator();

        while (expectedEnumerator.MoveNext())
        {
            bool moved = actualEnumerator.MoveNext();
            Debug.Assert(moved);

            var x = actualEnumerator.Current;
            var y = expectedEnumerator.Current;
            var v = y == T.One ? -T.Log(x) : -T.Log(T.One - x);
            cost += T.IsNaN(v) ? T.Zero : v;
        }

        return cost;
    }

    /// <inheritdoc />
    public static Tensor<T> Derivative(in ReadOnlyTensorSpan<T> expected, in ReadOnlyTensorSpan<T> actual)
    {
        if (!expected.Lengths.SequenceEqual(actual.Lengths))
            throw new ArgumentException(message: "Lengths of both input tensors don't match.");

        var outputs            = Tensor.Create<T>(actual.Lengths);
        var expectedEnumerator = expected.GetEnumerator();
        var actualEnumerator   = actual.GetEnumerator();
        var outputsEnumerator  = outputs.AsTensorSpan().GetEnumerator();

        while (expectedEnumerator.MoveNext())
        {
            bool actualMoved = actualEnumerator.MoveNext();
            Debug.Assert(actualMoved);
            bool outputsMoved = outputsEnumerator.MoveNext();
            Debug.Assert(outputsMoved);

            var x = actualEnumerator.Current;
            var y = expectedEnumerator.Current;
            if (x == T.Zero || x == T.One)
                outputsEnumerator.Current = T.Zero;
            else
                outputsEnumerator.Current = (-x + y) / (x * (x - T.One));
        }

        return outputs;
    }
}
