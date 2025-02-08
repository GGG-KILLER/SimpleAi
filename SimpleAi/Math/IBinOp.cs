using System.Numerics;

namespace SimpleAi.Math;

// Interface heavily inspired by Microsoft's Tensor library

internal interface IBinOp<T>
{
    static abstract T Execute(T left, T right);
    static abstract Vector<T> Execute(Vector<T> lefts, Vector<T> rights);
}

internal readonly partial struct AddOp<T> : IBinOp<T>
    where T : IAdditiveIdentity<T, T>, IAdditionOperators<T, T, T>
{
    public static T Execute(T left, T right) => left + right;
    public static Vector<T> Execute(Vector<T> lefts, Vector<T> rights) => lefts + rights;
}

internal readonly struct MulOp<T> : IBinOp<T>
    where T : IMultiplicativeIdentity<T, T>, IMultiplyOperators<T, T, T>
{
    public static T Execute(T left, T right) => left * right;
    public static Vector<T> Execute(Vector<T> lefts, Vector<T> rights) => lefts * rights;
}

internal readonly struct BUPipeline<T, TBin, TUn> : IBinOp<T>
    where TBin : struct, IBinOp<T>
    where TUn : struct, IUnOp<T>
{
    public static T Execute(T left, T right) => TUn.Execute(TBin.Execute(left, right));
    public static Vector<T> Execute(Vector<T> lefts, Vector<T> rights) =>
        TUn.Execute(TBin.Execute(lefts, rights));
}
