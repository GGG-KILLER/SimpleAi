using System;
using System.Globalization;
using System.Text.RegularExpressions;
using Avalonia.Data;
using Avalonia.Data.Converters;

namespace SimpleAi.UI.Views.Converters;

internal class AreaConverter : IValueConverter
{
    /// <inheritdoc />
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is ValueTuple<VectorTypeT, VectorTypeT> (var start, var end)
            && targetType.IsAssignableFrom(typeof(string)))
            return string.Format(
                CultureInfo.InvariantCulture,
                format: "{0},{1}:{2},{3}",
                start.X,
                start.Y,
                end.X,
                end.Y);

        return new BindingNotification(new InvalidCastException(), BindingErrorType.Error);
    }

    /// <inheritdoc />
    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is not string from || !targetType.IsAssignableFrom(typeof(ValueTuple<VectorTypeT, VectorTypeT>)))
            return new BindingNotification(new InvalidCastException(), BindingErrorType.Error);

        Match match = Regexes.Range.Match(from);
        if (!match.Success)
            return new BindingNotification(
                new FormatException(message: "Invalid area format, should be startX,startY:endX,endY"),
                BindingErrorType.Error);

        VectorTypeT start =
            (NumberTypeT.Parse(match.Groups[groupname: "startX"].ValueSpan, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture),
             NumberTypeT.Parse(
                 match.Groups[groupname: "startY"].ValueSpan,
                 NumberStyles.AllowDecimalPoint,
                 CultureInfo.InvariantCulture));
        VectorTypeT end =
            (NumberTypeT.Parse(match.Groups[groupname: "endX"].ValueSpan, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture),
             NumberTypeT.Parse(
                 match.Groups[groupname: "endY"].ValueSpan,
                 NumberStyles.AllowDecimalPoint,
                 CultureInfo.InvariantCulture));
        return (start, end);
    }
}
