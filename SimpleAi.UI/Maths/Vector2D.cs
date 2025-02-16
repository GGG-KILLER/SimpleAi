using MIConvexHull;

namespace SimpleAi.UI.Maths;

public readonly record struct Vector2D(NumberTypeT X, NumberTypeT Y) : IVertex2D
{
    double IVertex2D.X => X;

    double IVertex2D.Y => Y;
}
