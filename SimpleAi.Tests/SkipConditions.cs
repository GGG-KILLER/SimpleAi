using System.Numerics;

namespace SimpleAi.Tests;

public static class SkipConditions
{
    public static bool IntVectorsAreHardwareAccelerated => Vector.IsHardwareAccelerated && Vector<int>.IsSupported;
    public static bool FloatVectorsAreHardwareAccelerated => Vector.IsHardwareAccelerated && Vector<float>.IsSupported;
    public static bool DoubleVectorsAreHardwareAccelerated => Vector.IsHardwareAccelerated && Vector<double>.IsSupported;
}
