using System.Collections;

namespace SimpleAi.Internal;

internal sealed class MemoryIterator<T> : IEnumerable<T>
{
    public ReadOnlyMemory<T> Memory { get; set; }

    /// <inheritdoc />
    public IEnumerator<T> GetEnumerator() => new MemoryEnumerator(Memory);

    /// <inheritdoc />
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    private struct MemoryEnumerator(ReadOnlyMemory<T> memory) : IEnumerator<T>
    {
        private int _index = -1;

        /// <inheritdoc />
        public bool MoveNext()
        {
            if (_index >= memory.Length - 1) return false;

            _index++;
            return true;
        }

        /// <inheritdoc />
        public void Reset()
        {
            _index = -1;
        }

        /// <inheritdoc />
        public readonly T Current
            => _index >= 0
                   ? memory.Span[_index]
                   : throw new InvalidOperationException("Enumerator has not been initialized.");

        /// <inheritdoc />
        readonly object? IEnumerator.Current => Current;

        /// <inheritdoc />
        public readonly void Dispose() { }
    }
}
