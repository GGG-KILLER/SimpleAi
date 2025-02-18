using System.Numerics;

namespace SimpleAi.Math;

// Interface heavily inspired by Microsoft's Tensor library

internal interface ITernaryOp<T>
{
    static abstract T Execute(T left, T middle, T right);

    static abstract Vector<T> Execute(Vector<T> lefts, Vector<T> middles, Vector<T> rights);
}

/// <summary>
/// Combines two binary operations resulting in TSecond(TFirst(left, middle), right).
/// </summary>
internal readonly struct DoubleBinaryOp<T, TFirst, TSecond> : ITernaryOp<T>
    where TFirst : IBinOp<T> where TSecond : IBinOp<T>
{
    /// <inheritdoc />
    public static T Execute(T left, T middle, T right) => TSecond.Execute(TFirst.Execute(left, middle), right);

    /// <inheritdoc />
    public static Vector<T> Execute(Vector<T> lefts, Vector<T> middles, Vector<T> rights)
        => TSecond.Execute(TFirst.Execute(lefts, middles), rights);
}
