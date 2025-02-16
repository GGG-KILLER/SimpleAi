namespace SimpleAi.UI.Maths;

public readonly record struct Vector2DRange(Vector2D Start, Vector2D End)
{
    public NumberTypeT Width => End.X - Start.X;

    public NumberTypeT Height => End.Y - Start.Y;

    public bool IsInArea(NumberTypeT x, NumberTypeT y) => IsInArea(new Vector2D(x, y));

    public bool IsInArea(Vector2D vector)
        => Start.X <= vector.X && vector.X <= End.X && Start.Y <= vector.Y && vector.Y <= End.Y;
}
