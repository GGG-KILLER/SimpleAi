using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SimpleAi.Math;

// Interface heavily inspired by Microsoft's Tensor library

public interface IUnaryOp<T>
{
    static abstract T Execute(T arg);
    static abstract Vector<T> Execute(Vector<T> args);
}

public interface IBinaryOp<T>
{
    static abstract T Execute(T left, T right);
    static abstract Vector<T> Execute(Vector<T> lefts, Vector<T> rights);
}

public interface ITernaryOp<T>
{
    static abstract T Execute(T left, T middle, T right);
    static abstract Vector<T> Execute(Vector<T> lefts, Vector<T> middles, Vector<T> rights);
}

public interface IAggregateOp<T> : IBinaryOp<T>
{
    static abstract T InitialScalar { get; }
    static abstract Vector<T> InitialVector { get; }

    static abstract T Execute(Vector<T> args);
}

public readonly struct ReLUOp<T> : IUnaryOp<T>
    where T : INumber<T>
{
    public static T Execute(T arg) => T.Max(T.Zero, arg);
    public static Vector<T> Execute(Vector<T> args) => Vector.Max(Vector<T>.Zero, args);
}

public readonly struct AddOp<T> : IAggregateOp<T>
    where T : IAdditiveIdentity<T, T>, IAdditionOperators<T, T, T>
{
    public static T InitialScalar => T.AdditiveIdentity;
    public static Vector<T> InitialVector => Vector<T>.Zero;

    public static T Execute(Vector<T> args) => Vector.Sum(args);
    public static T Execute(T left, T right) => left + right;
    public static Vector<T> Execute(Vector<T> lefts, Vector<T> rights) => lefts + rights;
}

public readonly struct MulOp<T> : IAggregateOp<T>
    where T : IMultiplicativeIdentity<T, T>, IMultiplyOperators<T, T, T>
{
    public static T InitialScalar => T.MultiplicativeIdentity;
    public static Vector<T> InitialVector => Vector<T>.One;

    public static T Execute(Vector<T> args)
    {
        var acc = T.MultiplicativeIdentity;
        for (var idx = 0; idx < Vector<T>.Count; idx++)
            acc *= args.GetElement(idx);
        return acc;
    }
    public static T Execute(T left, T right) => left * right;
    public static Vector<T> Execute(Vector<T> lefts, Vector<T> rights) => lefts * rights;
}

public readonly struct BUPipeline<T, TBin, TUn> : IBinaryOp<T>
    where TBin : IBinaryOp<T>
    where TUn : IUnaryOp<T>
{
    public static T Execute(T left, T right) => TUn.Execute(TBin.Execute(left, right));

    public static Vector<T> Execute(Vector<T> lefts, Vector<T> rights) =>
        TUn.Execute(TBin.Execute(lefts, rights));
}

internal static partial class MathEx
{
    private static bool InputOutputSpanNonOverlapping<T>(ReadOnlySpan<T> input, Span<T> output)
    {
        return Unsafe.AreSame(ref input.Ref(), ref output.Ref()) || !input.Overlaps(output);
    }

    public static void Unary<T, TOp>(ReadOnlySpan<T> inputs, Span<T> outputs)
        where TOp : IUnaryOp<T>
    {
        Debug.Assert(inputs.Length <= outputs.Length);
        Debug.Assert(InputOutputSpanNonOverlapping(inputs, outputs));

        var idx = 0;

        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported && inputs.Length > Vector<T>.Count)
        {
            for (; idx < inputs.Length - Vector<T>.Count; idx += Vector<T>.Count)
            {
                var input = Vector.LoadUnsafe(ref inputs.UnsafeIndex(idx));
                TOp.Execute(input).StoreUnsafe(ref outputs.UnsafeIndex(idx));
            }
        }

        for (; idx < inputs.Length; idx++)
        {
            outputs.UnsafeIndex(idx) = TOp.Execute(inputs.UnsafeIndex(idx));
        }
    }

    public static void Binary<T, TOp>(ReadOnlySpan<T> lefts, ReadOnlySpan<T> rights, Span<T> outputs)
        where TOp : IBinaryOp<T>
    {
        Debug.Assert(lefts.Length == rights.Length, "Both inputs must be the same size.");
        Debug.Assert(lefts.Length <= outputs.Length, "Output must have enough space to store results.");
        Debug.Assert(InputOutputSpanNonOverlapping(lefts, outputs));
        Debug.Assert(InputOutputSpanNonOverlapping(rights, outputs));

        var idx = 0;

        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported && lefts.Length > Vector<T>.Count)
        {
            for (; idx < lefts.Length - Vector<T>.Count; idx += Vector<T>.Count)
            {
                var left = Vector.LoadUnsafe(ref lefts.UnsafeIndex(idx));
                var right = Vector.LoadUnsafe(ref rights.UnsafeIndex(idx));
                TOp.Execute(left, right).StoreUnsafe(ref outputs.UnsafeIndex(idx));
            }
        }

        for (; idx < lefts.Length; idx++)
        {
            outputs.UnsafeIndex(idx) = TOp.Execute(lefts.UnsafeIndex(idx), rights.UnsafeIndex(idx));
        }
    }

    public static void Ternary<T, TOp>(
        ReadOnlySpan<T> lefts,
        ReadOnlySpan<T> middles,
        ReadOnlySpan<T> rights,
        Span<T> outputs)
        where TOp : ITernaryOp<T>
    {
        Debug.Assert(lefts.Length == rights.Length && middles.Length == rights.Length, "All inputs must be the same size.");
        Debug.Assert(lefts.Length <= outputs.Length, "Output must have enough space to store results.");
        Debug.Assert(InputOutputSpanNonOverlapping(lefts, outputs));
        Debug.Assert(InputOutputSpanNonOverlapping(middles, outputs));
        Debug.Assert(InputOutputSpanNonOverlapping(rights, outputs));

        var idx = 0;

        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported && lefts.Length > Vector<T>.Count)
        {
            for (; idx < lefts.Length - Vector<T>.Count; idx += Vector<T>.Count)
            {
                var left = Vector.LoadUnsafe(ref lefts.UnsafeIndex(idx));
                var middle = Vector.LoadUnsafe(ref middles.UnsafeIndex(idx));
                var right = Vector.LoadUnsafe(ref rights.UnsafeIndex(idx));
                TOp.Execute(left, middle, right).StoreUnsafe(ref outputs.UnsafeIndex(idx));
            }
        }

        for (; idx < lefts.Length; idx++)
        {
            outputs.UnsafeIndex(idx) = TOp.Execute(
                lefts.UnsafeIndex(idx),
                middles.UnsafeIndex(idx),
                rights.UnsafeIndex(idx));
        }
    }

    public static T Aggregate<T, TUnOp, TAggOp>(ReadOnlySpan<T> inputs)
        where TUnOp : IUnaryOp<T>
        where TAggOp : IAggregateOp<T>
    {
        var idx = 0;
        var acc = TAggOp.InitialScalar;

        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported && inputs.Length > Vector<T>.Count)
        {
            var vecAcc = TAggOp.InitialVector;

            for (; idx < inputs.Length - Vector<T>.Count; idx += Vector<T>.Count)
            {
                var input = Vector.LoadUnsafe(ref inputs.UnsafeIndex(idx));
                vecAcc = TAggOp.Execute(vecAcc, TUnOp.Execute(input));
            }

            acc = TAggOp.Execute(vecAcc);
        }

        for (; idx < inputs.Length; idx++)
        {
            acc = TAggOp.Execute(acc, TUnOp.Execute(inputs.UnsafeIndex(idx)));
        }

        return acc;
    }

    public static T Aggregate<T, TBinOp, TAggOp>(ReadOnlySpan<T> lefts, ReadOnlySpan<T> rights)
        where TBinOp : IBinaryOp<T>
        where TAggOp : IAggregateOp<T>
    {
        Debug.Assert(lefts.Length == rights.Length, "Both inputs must be the same size.");

        var idx = 0;
        var acc = TAggOp.InitialScalar;

        if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported && lefts.Length > Vector<T>.Count)
        {
            var vecAcc = TAggOp.InitialVector;

            for (; idx < lefts.Length - Vector<T>.Count; idx += Vector<T>.Count)
            {
                var left = Vector.LoadUnsafe(ref lefts.UnsafeIndex(idx));
                var right = Vector.LoadUnsafe(ref rights.UnsafeIndex(idx));
                vecAcc = TAggOp.Execute(vecAcc, TBinOp.Execute(left, right));
            }

            acc = TAggOp.Execute(vecAcc);
        }

        for (; idx < lefts.Length; idx++)
        {
            var left = lefts.UnsafeIndex(idx);
            var right = rights.UnsafeIndex(idx);
            acc = TAggOp.Execute(acc, TBinOp.Execute(left, right));
        }

        return acc;
    }
}
