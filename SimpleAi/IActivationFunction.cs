using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using JetBrains.Annotations;
using SimpleAi.Math;

namespace SimpleAi;

/// <summary>
/// The interface for activation functions.
/// </summary>
/// <typeparam name="T">The numeric type accepted by the activation function.</typeparam>
[PublicAPI]
public interface IActivationFunction<T>
{
    /// <summary>
    /// Executes the activation function on a set of <paramref name="inputs"/>, storing its result in the
    /// <paramref name="outputs"/> buffer.
    /// </summary>
    /// <param name="inputs">The inputs to the activation function.</param>
    /// <param name="outputs">The buffer to store the outputs for each input on.</param>
    [PublicAPI]
    static abstract void Activate(ReadOnlySpan<T> inputs, Span<T> outputs);

    /// <summary>
    /// Runs the provided input <paramref name="values"/> through the derivative of the activation function, storing its
    /// results in the <paramref name="outputs"/> buffer.
    /// </summary>
    /// <param name="values">The values to pass through the derivative of the activation function.</param>
    /// <param name="outputs">The buffer where the results of the derivative will be stored.</param>
    [PublicAPI]
    static abstract void Derivative(ReadOnlySpan<T> values, Span<T> outputs);
}

[PublicAPI]
public readonly struct Sigmoid<T> : IActivationFunction<T>
    where T : INumberBase<T> /* T.One */, IExponentialFunctions<T> // T.Exp
{
    /// <inheritdoc />
    [PublicAPI]
    public static void Activate(ReadOnlySpan<T> inputs, Span<T> outputs)
        => MathEx.Unary<T, ActivationOp>(inputs, outputs);

    /// <inheritdoc />
    [PublicAPI]
    public static void Derivative(ReadOnlySpan<T> inputs, Span<T> outputs)
        => MathEx.Unary<T, DerivativeOp>(inputs, outputs);

    private readonly struct ActivationOp : IUnOp<T>
    {
        public static T Execute(T value) => T.One / (T.One + T.Exp(-value));

        public static Vector<T> Execute(Vector<T> values)
        {
            Vector<T> exp = ExpOp<T>.Execute(-values);
            return Vector<T>.One / (Vector<T>.One + exp);
        }
    }

    private readonly struct DerivativeOp : IUnOp<T>
    {
        public static T Execute(T value)
        {
            T activated = ActivationOp.Execute(value);
            return activated * (T.One - activated);
        }

        public static Vector<T> Execute(Vector<T> values)
        {
            Vector<T> activated = ActivationOp.Execute(values);
            return activated * (Vector<T>.One - activated);
        }
    }
}

[PublicAPI]
public readonly struct TanH<T> : IActivationFunction<T>
    where T : INumberBase<T> /* T.One */, IExponentialFunctions<T> // T.Exp
{
    /// <inheritdoc />
    [PublicAPI]
    public static void Activate(ReadOnlySpan<T> inputs, Span<T> outputs)
        => MathEx.Unary<T, ActivationOp>(inputs, outputs);

    /// <inheritdoc />
    [PublicAPI]
    public static void Derivative(ReadOnlySpan<T> inputs, Span<T> outputs)
        => MathEx.Unary<T, DerivativeOp>(inputs, outputs);

    private readonly struct ActivationOp : IUnOp<T>
    {
        public static T Execute(T value)
        {
            T exp2 = T.Exp((T.One + T.One) * value);
            return (exp2 - T.One) / (exp2 + T.One);
        }

        public static Vector<T> Execute(Vector<T> values)
        {
            Vector<T> exp2 = ExpOp<T>.Execute(Vector.Create(T.One + T.One) * values);
            return (exp2 - Vector<T>.One) / (exp2 + Vector<T>.One);
        }
    }

    private readonly struct DerivativeOp : IUnOp<T>
    {
        public static T Execute(T value)
        {
            T activated = ActivationOp.Execute(value);
            return T.One - (activated * activated);
        }

        public static Vector<T> Execute(Vector<T> values)
        {
            Vector<T> activated = ActivationOp.Execute(values);
            return Vector<T>.One - (activated * activated);
        }
    }
}

[PublicAPI]
[SuppressMessage(category: "ReSharper", checkId: "InconsistentNaming", Justification = "Name of algorithm.")]
public readonly struct ReLU<T> : IActivationFunction<T> where T : INumber<T> // T.Zero, T.Max
{
    /// <inheritdoc />
    [PublicAPI]
    public static void Activate(ReadOnlySpan<T> inputs, Span<T> outputs)
        => MathEx.Unary<T, ActivationOp>(inputs, outputs);

    /// <inheritdoc />
    [PublicAPI]
    public static void Derivative(ReadOnlySpan<T> inputs, Span<T> outputs)
        => MathEx.Unary<T, DerivativeOp>(inputs, outputs);

    private readonly struct ActivationOp : IUnOp<T>
    {
        public static T Execute(T value) => T.Max(T.Zero, value);

        public static Vector<T> Execute(Vector<T> values) => Vector.Max(Vector<T>.Zero, values);
    }

    private readonly struct DerivativeOp : IUnOp<T>
    {
        public static T Execute(T value) => value > T.Zero ? T.One : T.Zero;

        public static Vector<T> Execute(Vector<T> values)
            => Vector.ConditionalSelect(Vector.GreaterThan(values, Vector<T>.Zero), Vector<T>.One, Vector<T>.Zero);
    }
}

[PublicAPI]
public readonly struct SoftMax<T> : IActivationFunction<T>
    where T : IExponentialFunctions<T>,                   // T.Exp
    IAdditiveIdentity<T, T>, IAdditionOperators<T, T, T>, // AddOp<T>
    IDivisionOperators<T, T, T>,                          // DivOp<T>, LastDerivativeStep (operator /)
    ISubtractionOperators<T, T, T>,                       // LastDerivativeStep (operator -)
    IMultiplyOperators<T, T, T>                           // LastDerivativeStep (operator *)
{
    /// <inheritdoc />
    [PublicAPI]
    public static void Activate(ReadOnlySpan<T> inputs, Span<T> outputs)
    {
        MathEx.Unary<T, ExpOp<T>>(inputs, outputs);
        T expSum = MathEx.Aggregate<T, Identity<T>, AddOp<T>>(outputs);
        MathEx.Binary<T, DivOp<T>>(outputs, expSum, outputs);
    }

    /// <inheritdoc />
    [PublicAPI]
    public static void Derivative(ReadOnlySpan<T> inputs, Span<T> outputs)
    {
        MathEx.Unary<T, ExpOp<T>>(inputs, outputs);
        T expSum = MathEx.Aggregate<T, Identity<T>, AddOp<T>>(outputs);
        MathEx.Binary<T, LastDerivativeStep>(outputs, expSum, outputs);
    }

    private readonly struct LastDerivativeStep : IBinOp<T>
    {
        public static T Execute(T exp, T expSum) => ((exp * expSum) - (exp * exp)) / (expSum * expSum);

        public static Vector<T> Execute(Vector<T> exp, Vector<T> expSum)
            => ((exp * expSum) - (exp * exp)) / (expSum * expSum);
    }
}
