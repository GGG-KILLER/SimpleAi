using System.Numerics;

namespace SimpleAi.Math;

// Interface heavily inspired by Microsoft's Tensor library

internal interface IUnOp<T>
{
    static abstract T Execute(T arg);

    static abstract Vector<T> Execute(Vector<T> args);
}

internal readonly struct Identity<T> : IUnOp<T>
{
    public static T Execute(T arg) => arg;

    public static Vector<T> Execute(Vector<T> args) => args;
}

internal readonly struct ExpOp<T> : IUnOp<T> where T : IExponentialFunctions<T>
{
    public static T Execute(T arg) => T.Exp(arg);

    public static Vector<T> Execute(Vector<T> args)
    {
        if (typeof(T) == typeof(float)) return Vector.Exp(args.As<T, float>()).As<float, T>();
        if (typeof(T) == typeof(double)) return Vector.Exp(args.As<T, double>()).As<double, T>();
        Vector<T> output                                       = Vector<T>.Zero;
        for (var idx = 0; idx < Vector<T>.Count; idx++) output = output.WithElement(idx, T.Exp(args[idx]));
        return output;
    }
}

internal readonly struct Pow2Op<T> : IUnOp<T> where T : IMultiplyOperators<T, T, T>
{
    public static T Execute(T arg) => arg * arg;

    public static Vector<T> Execute(Vector<T> args) => args * args;
}
