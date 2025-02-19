using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SimpleAi;

internal static class SpanExtensions
{
    public static ref T Ref<T>(this T[] array) => ref MemoryMarshal.GetArrayDataReference(array);

    public static ref T Ref<T>(this Span<T> span) => ref MemoryMarshal.GetReference(span);

    public static ref T Ref<T>(this ReadOnlySpan<T> span) => ref MemoryMarshal.GetReference(span);

    public static ref T UnsafeIndex<T>(this T[] array, int offset)
    {
        Debug.Assert(array.Length >= offset, "Attempt to do out of bounds indexing.");
        return ref Unsafe.Add(ref array.Ref(), offset);
    }

    public static ref T UnsafeIndex<T>(this T[] array, Index index)
        => ref array.UnsafeIndex(index.GetOffset(array.Length));

    public static ref T UnsafeIndex<T>(this Span<T> span, int offset)
    {
        Debug.Assert(span.Length >= offset, "Attempt to do out of bounds indexing.");
        return ref Unsafe.Add(ref span.Ref(), offset);
    }

    public static ref T UnsafeIndex<T>(this Span<T> span, Index index)
        => ref span.UnsafeIndex(index.GetOffset(span.Length));

    public static ref T UnsafeIndex<T>(this ReadOnlySpan<T> span, int offset)
    {
        Debug.Assert(span.Length >= offset, "Attempt to do out of bounds indexing.");
        return ref Unsafe.Add(ref span.Ref(), offset);
    }

    public static ref T UnsafeIndex<T>(this ReadOnlySpan<T> span, Index index)
        => ref span.UnsafeIndex(index.GetOffset(span.Length));
}
