using System.Numerics;

namespace SimpleAi.Math;

// Interface heavily inspired by Microsoft's Tensor library

internal interface IAggOp<T> : IBinOp<T>
{
    static abstract T InitialScalar { get; }
    static abstract Vector<T> InitialVector { get; }

    static abstract T Execute(Vector<T> args);
}

internal readonly partial struct AddOp<T> : IAggOp<T>
    where T : IAdditiveIdentity<T, T>, IAdditionOperators<T, T, T>
{
    public static T InitialScalar => T.AdditiveIdentity;
    public static Vector<T> InitialVector => Vector<T>.Zero;

    public static T Execute(Vector<T> args) => Vector.Sum(args);
}
