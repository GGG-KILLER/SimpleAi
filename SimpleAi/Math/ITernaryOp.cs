using System.Numerics;

namespace SimpleAi.Math;

// Interface heavily inspired by Microsoft's Tensor library

internal interface ITernaryOp<T>
{
    static abstract T Execute(T left, T middle, T right);

    static abstract Vector<T> Execute(Vector<T> lefts, Vector<T> middles, Vector<T> rights);
}
