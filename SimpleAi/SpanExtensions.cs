using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SimpleAi;

internal static class SpanExtensions
{
    public static ref T Ref<T>(this T[] array) =>
        ref MemoryMarshal.GetArrayDataReference(array);

    public static ref T Ref<T>(this Span<T> span) =>
        ref MemoryMarshal.GetReference(span);

    public static ref T Ref<T>(this ReadOnlySpan<T> span) =>
        ref MemoryMarshal.GetReference(span);

    public static ref T UnsafeIndex<T>(this T[] array, int offset) =>
        ref Unsafe.Add(ref array.Ref(), offset);

    public static ref T UnsafeIndex<T>(this Span<T> span, int offset) =>
        ref Unsafe.Add(ref span.Ref(), offset);

    public static ref T UnsafeIndex<T>(this ReadOnlySpan<T> span, int offset) =>
        ref Unsafe.Add(ref span.Ref(), offset);
}
