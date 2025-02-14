using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using SimpleAi.Math;

namespace SimpleAi;

public interface IActivationFunction<T>
{
    static abstract void Activate(ReadOnlySpan<T> inputs, Span<T> outputs);

    static abstract void Derivative(ReadOnlySpan<T> values, Span<T> outputs);
}

public readonly struct Sigmoid<T> : IActivationFunction<T>
    where T : INumberBase<T> /* T.One */, IExponentialFunctions<T> // T.Exp
{
    public static void Activate(ReadOnlySpan<T> inputs, Span<T> outputs)
        => MathEx.Unary<T, ActivationOp>(inputs, outputs);

    public static void Derivative(ReadOnlySpan<T> inputs, Span<T> outputs)
        => MathEx.Unary<T, DerivativeOp>(inputs, outputs);

    private readonly struct ActivationOp : IUnOp<T>
    {
        public static T Execute(T value) => T.One / (T.One + T.Exp(value));

        public static Vector<T> Execute(Vector<T> values)
        {
            var exp = ExpOp<T>.Execute(values);
            return Vector<T>.One / (Vector<T>.One + exp);
        }
    }

    private readonly struct DerivativeOp : IUnOp<T>
    {
        public static T Execute(T value)
        {
            var activated = ActivationOp.Execute(value);
            return activated * (T.One - activated);
        }

        public static Vector<T> Execute(Vector<T> values)
        {
            var activated = ActivationOp.Execute(values);
            return activated * (Vector<T>.One - activated);
        }
    }
}

public readonly struct TanH<T> : IActivationFunction<T>
    where T : INumberBase<T> /* T.One */, IExponentialFunctions<T> // T.Exp
{
    public static void Activate(ReadOnlySpan<T> inputs, Span<T> outputs)
        => MathEx.Unary<T, ActivationOp>(inputs, outputs);

    public static void Derivative(ReadOnlySpan<T> inputs, Span<T> outputs)
        => MathEx.Unary<T, DerivativeOp>(inputs, outputs);

    private readonly struct ActivationOp : IUnOp<T>
    {
        public static T Execute(T value)
        {
            var exp2 = T.Exp((T.One + T.One) * value);
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
            var activated = ActivationOp.Execute(value);
            return T.One - activated * activated;
        }

        public static Vector<T> Execute(Vector<T> values)
        {
            var activated = ActivationOp.Execute(values);
            return Vector<T>.One - activated * activated;
        }
    }
}

[SuppressMessage("ReSharper", "InconsistentNaming", Justification = "Name of algorithm.")]
public readonly struct ReLU<T> : IActivationFunction<T> where T : INumber<T> // T.Zero, T.Max
{
    public static void Activate(ReadOnlySpan<T> inputs, Span<T> outputs)
        => MathEx.Unary<T, ActivationOp>(inputs, outputs);

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

public readonly struct SoftMax<T> : IActivationFunction<T>
    where T : IExponentialFunctions<T>,                   // T.Exp
    IAdditiveIdentity<T, T>, IAdditionOperators<T, T, T>, // AddOp<T>
    IDivisionOperators<T, T, T>,                          // DivOp<T>, LastDerivativeStep (operator /)
    ISubtractionOperators<T, T, T>,                       // LastDerivativeStep (operator -)
    IMultiplyOperators<T, T, T>                           // LastDerivativeStep (operator *)
{
    public static void Activate(ReadOnlySpan<T> inputs, Span<T> outputs)
    {
        MathEx.Unary<T, ExpOp<T>>(inputs, outputs);
        var expSum = MathEx.Aggregate<T, Identity<T>, AddOp<T>>(outputs);
        MathEx.Binary<T, DivOp<T>>(outputs, expSum, outputs);
    }

    public static void Derivative(ReadOnlySpan<T> inputs, Span<T> outputs)
    {
        MathEx.Unary<T, ExpOp<T>>(inputs, outputs);
        var expSum = MathEx.Aggregate<T, Identity<T>, AddOp<T>>(outputs);
        MathEx.Binary<T, LastDerivativeStep>(outputs, expSum, outputs);
    }

    private readonly struct LastDerivativeStep : IBinOp<T>
    {
        public static T Execute(T exp, T expSum) => (exp * expSum - exp * exp) / (expSum * expSum);

        public static Vector<T> Execute(Vector<T> exp, Vector<T> expSum)
            => (exp * expSum - exp * exp) / (expSum * expSum);
    }
}
