using System.Numerics;
using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi;

/// <summary>The interface for loss functions.</summary>
/// <typeparam name="T">The numeric type accepted by the loss function.</typeparam>
[PublicAPI]
public interface ILossFunction<T>
{
    /// <summary>
    ///     Calculates the loss of the neural network based on the <paramref name="expected" /> outputs and the
    ///     <paramref name="actual" /> inputs.
    /// </summary>
    /// <param name="expected">The inputs that were expected to be returned by the neural network.</param>
    /// <param name="actual">The inputs that were actually obtained when executing the inference.</param>
    /// <returns>The current loss of the neural network.</returns>
    [PublicAPI]
    static abstract T Calculate(in ReadOnlyTensorSpan<T> expected, in ReadOnlyTensorSpan<T> actual);

    /// <summary>
    ///     Calculates the derivative of the loss function at the provided <paramref name="expected" /> and
    ///     <paramref name="actual" /> inputs.
    /// </summary>
    /// <param name="expected">The inputs that were expected to be returned by the neural network.</param>
    /// <param name="actual">The inputs that were actually obtained when executing the inference.</param>
    /// <returns>The results of executing the derivative.</returns>
    [PublicAPI]
    static abstract Tensor<T> Derivative(in ReadOnlyTensorSpan<T> expected, in ReadOnlyTensorSpan<T> actual);
}

[PublicAPI]
public readonly struct MeanSquaredError<T> : ILossFunction<T>
    where T : ISubtractionOperators<T, T, T>,                   // Tensor.Subtract
    IMultiplyOperators<T, T, T>, IMultiplicativeIdentity<T, T>, // Tensor.Multiply
    IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>,       // Tensor.Sum
    INumberBase<T>,                                             // T.CreateSaturating
    IDivisionOperators<T, T, T>                                 // operator /
{
    /// <inheritdoc />
    public static T Calculate(in ReadOnlyTensorSpan<T> expected, in ReadOnlyTensorSpan<T> actual)
    {
        if (!expected.Lengths.SequenceEqual(actual.Lengths))
            throw new ArgumentException(message: "Lengths of both input tensors don't match.");

        Tensor<T> error           = Tensor.Subtract(actual, expected);
        T         squaredErrorSum = Tensor.Sum<T>(Tensor.Multiply<T>(error, error));
        return squaredErrorSum / T.CreateSaturating(actual.FlattenedLength);
    }

    /// <inheritdoc />
    public static Tensor<T> Derivative(in ReadOnlyTensorSpan<T> expected, in ReadOnlyTensorSpan<T> actual)
    {
        if (!expected.Lengths.SequenceEqual(actual.Lengths))
            throw new ArgumentException(message: "Lengths of both input tensors don't match.");

        return Tensor.Divide(
            Tensor.Multiply(Tensor.Subtract(actual, expected), T.CreateSaturating(2)),
            T.CreateSaturating(actual.FlattenedLength));
    }
}

[PublicAPI]
public readonly struct BinaryCrossEntropy<T> : ILossFunction<T>
    where T : IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>, // T.AdditiveIdentity, operator +
    ILogarithmicFunctions<T>,                                       // T.Log
    INumber<T>,                                                     // T.One, T.Zero, T.IsNaN, T.Max
    IComparisonOperators<T, T, bool>                                // operator >
{
    /// <inheritdoc />
    public static T Calculate(in ReadOnlyTensorSpan<T> y, in ReadOnlyTensorSpan<T> a)
    {
        if (!y.Lengths.SequenceEqual(a.Lengths))
            throw new ArgumentException("Lengths of both input tensors don't match.");

        T         epsilon   = T.CreateChecked(1e-9);            // Small value to prevent log(0)
        Tensor<T> clampedA  = Tensor.Max(a, epsilon);           // Clamp to prevent log(0)
        Tensor<T> oneMinusA = Tensor.Subtract(T.One, clampedA); // 1 - a

        // BCE formula: -[y * log(a) + (1 - y) * log(1 - a)]
        Tensor<T> logA         = Tensor.Log<T>(clampedA);
        Tensor<T> logOneMinusA = Tensor.Log<T>(oneMinusA);
        Tensor<T> firstTerm    = Tensor.Multiply(y, logA);                           // y * log(a)
        Tensor<T> secondTerm   = Tensor.Multiply<T>(Tensor.Negate(y), logOneMinusA); // (1 - y) * log(1 - a)

        Tensor<T> losses = Tensor.Add<T>(firstTerm, secondTerm); // Sum of both terms
        return Tensor.Sum<T>(losses);                            // Sum across all samples
    }

    /// <inheritdoc />
    public static Tensor<T> Derivative(in ReadOnlyTensorSpan<T> y, in ReadOnlyTensorSpan<T> a)
    {
        if (!y.Lengths.SequenceEqual(a.Lengths))
            throw new ArgumentException("Lengths of both input tensors don't match.");

        // Derivative: (a - y) / (a * (1 - a))
        Tensor<T> oneMinusA = Tensor.Subtract(T.One, a); // 1 - a

        // (a - y) / (a * (1 - a))
        return Tensor.Divide<T>(Tensor.Subtract(a, y), Tensor.Multiply(a, oneMinusA));
    }
}

[PublicAPI]
public readonly struct MultiClassCrossEntropy<T> : ILossFunction<T>
    where T : IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>, // T.AdditiveIdentity, operator +
    ILogarithmicFunctions<T>,                                       // T.Log
    INumber<T>,                                                     // T.One, T.Zero, T.IsNaN, T.Max
    IComparisonOperators<T, T, bool>                                // operator >
{
    /// <inheritdoc />
    public static T Calculate(in ReadOnlyTensorSpan<T> y, in ReadOnlyTensorSpan<T> a)
    {
        if (!y.Lengths.SequenceEqual(a.Lengths))
            throw new ArgumentException("Lengths of both input tensors don't match.");

        T         epsilon  = T.CreateChecked(1e-9);  // Small value to prevent log(0)
        Tensor<T> clampedA = Tensor.Max(a, epsilon); // Clamp to prevent log(0)
        Tensor<T> negY     = Tensor.Negate(y);
        Tensor<T> logA     = Tensor.Log<T>(clampedA);
        Tensor<T> losses   = Tensor.Multiply<T>(negY, logA);
        return Tensor.Sum<T>(losses);
    }

    /// <inheritdoc />
    public static Tensor<T> Derivative(in ReadOnlyTensorSpan<T> y, in ReadOnlyTensorSpan<T> a)
    {
        if (!y.Lengths.SequenceEqual(a.Lengths))
            throw new ArgumentException("Lengths of both input tensors don't match.");

        // -y / a
        return Tensor.Divide(Tensor.Negate(y), a);
    }
}
