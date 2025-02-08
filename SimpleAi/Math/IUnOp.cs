using System.Numerics;

namespace SimpleAi.Math;

// Interface heavily inspired by Microsoft's Tensor library

internal interface IUnOp<T>
{
    static abstract T Execute(T arg);
    static abstract Vector<T> Execute(Vector<T> args);
}

internal readonly struct ReLUOp<T> : IUnOp<T>
    where T : INumber<T>
{
    public static T Execute(T arg) => T.Max(T.Zero, arg);
    public static Vector<T> Execute(Vector<T> args) => Vector.Max(Vector<T>.Zero, args);
}
