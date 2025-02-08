using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SimpleAi.Math;

// Interface heavily inspired by Microsoft's Tensor library

internal static partial class MathEx
{
    private static bool InputOutputSpanNonOverlapping<T>(ReadOnlySpan<T> input, Span<T> output)
    {
        return Unsafe.AreSame(ref input.Ref(), ref output.Ref()) || !input.Overlaps(output);
    }

    [SkipLocalsInit]
    public static void Unary<T, TOp>(ReadOnlySpan<T> inputs, Span<T> outputs)
        where TOp : struct, IUnOp<T>
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

    [SkipLocalsInit]
    public static void Binary<T, TOp>(ReadOnlySpan<T> lefts, ReadOnlySpan<T> rights, Span<T> outputs)
        where TOp : struct, IBinOp<T>
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

    [SkipLocalsInit]
    public static void Ternary<T, TOp>(
        ReadOnlySpan<T> lefts,
        ReadOnlySpan<T> middles,
        ReadOnlySpan<T> rights,
        Span<T> outputs)
        where TOp : struct, ITernaryOp<T>
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

    [SkipLocalsInit]
    public static T Aggregate<T, TUnOp, TAggOp>(ReadOnlySpan<T> inputs)
        where TUnOp : struct, IUnOp<T>
        where TAggOp : struct, IAggOp<T>
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

    [SkipLocalsInit]
    public static T Aggregate<T, TBinOp, TAggOp>(ReadOnlySpan<T> lefts, ReadOnlySpan<T> rights)
        where TBinOp : struct, IBinOp<T>
        where TAggOp : struct, IAggOp<T>
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
