using System;
using System.Globalization;
using System.Text.RegularExpressions;
using Avalonia.Data;
using Avalonia.Data.Converters;
using SimpleAi.UI.Maths;

namespace SimpleAi.UI.Views.Converters;

internal class Vector2DRangeConverter : IValueConverter
{
    /// <inheritdoc />
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is Vector2DRange range && targetType.IsAssignableFrom(typeof(string)))
            return string.Format(
                CultureInfo.InvariantCulture,
                format: "{0},{1}:{2},{3}",
                range.Start.X,
                range.Start.Y,
                range.End.X,
                range.End.Y);

        return new BindingNotification(new InvalidCastException(), BindingErrorType.Error);
    }

    /// <inheritdoc />
    public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is not string from || !targetType.IsAssignableFrom(typeof(Vector2DRange)))
            return new BindingNotification(new InvalidCastException(), BindingErrorType.Error);

        Match match = Regexes.Range.Match(from);
        if (!match.Success)
            return new BindingNotification(
                new FormatException(message: "Invalid area format, should be startX,startY:endX,endY"),
                BindingErrorType.Error);

        return new Vector2DRange(
            new Vector2D(
                NumberTypeT.Parse(
                    match.Groups[groupname: "startX"].ValueSpan,
                    NumberStyles.AllowDecimalPoint,
                    CultureInfo.InvariantCulture),
                NumberTypeT.Parse(
                    match.Groups[groupname: "startY"].ValueSpan,
                    NumberStyles.AllowDecimalPoint,
                    CultureInfo.InvariantCulture)),
            new Vector2D(
                NumberTypeT.Parse(
                    match.Groups[groupname: "endX"].ValueSpan,
                    NumberStyles.AllowDecimalPoint,
                    CultureInfo.InvariantCulture),
                NumberTypeT.Parse(
                    match.Groups[groupname: "endY"].ValueSpan,
                    NumberStyles.AllowDecimalPoint,
                    CultureInfo.InvariantCulture)));
    }
}
