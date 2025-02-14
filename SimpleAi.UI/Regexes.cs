using System.Text.RegularExpressions;

namespace SimpleAi.UI;

internal static partial class Regexes
{
    [GeneratedRegex(
        @"^\s*(?<startX>\d+(\.\d+)?)\s*,\s*(?<startY>\d+(\.\d+)?)\s*:\s*(?<endX>\d+(\.\d+)?)\s*,\s*(?<endY>\d+(\.\d+)?)\s*$",
        RegexOptions.CultureInvariant
        | RegexOptions.ExplicitCapture
        | RegexOptions.IgnoreCase
        | RegexOptions.Singleline)]
    public static partial Regex Range { get; }
}
