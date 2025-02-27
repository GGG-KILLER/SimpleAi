using System.Diagnostics;

namespace SimpleAi.Internal;

internal sealed class ObjectPool<T>(Func<T> factory, int maximumSize) where T : class
{
    private struct Box
    {
        internal T? Value;
    }

    private readonly Box[] _items = maximumSize > 2 ? new Box[maximumSize - 2] : [];
    private          T?    _first;
    private          T?    _second;

    public T Rent()
    {
        if (Interlocked.Exchange(ref _first, null) is { } first) return first;

        if (Interlocked.Exchange(ref _second, null) is { } second) return second;

        return RentSlow();
    }

    public void Return(T value)
    {
        Debug.Assert(value is not null);

        if (_first is null)
            _first = value;
        else if (_second is null)
            _second = value;
        else
            ReturnSlow(value);
    }

    private void ReturnSlow(T value)
    {
        var items = _items;
        for (var idx = 0; idx < items.Length; idx++)
        {
            if (items[idx].Value is null)
            {
                items[idx].Value = value;
                break;
            }
        }
    }

    private T RentSlow()
    {
        var items = _items;
        for (var idx = 0; idx < items.Length; idx++)
        {
            if (Interlocked.Exchange(ref items[idx].Value, null) is { } item) return item;
        }

        return factory();
    }
}
