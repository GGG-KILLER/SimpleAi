using ScottPlot;

namespace SimpleAi.UI.Plotting;

/// <summary>
/// Slide the view to the right to keep the newest data points in view
/// </summary>
internal class ConstantSlide : IAxisLimitManager
{
    /// <summary>
    /// Amount of horizontal area to display (in axis units)
    /// </summary>
    public double Width { get; set; } = 1000;

    /// <summary>
    /// Defines the amount of whitespace added to the right of the data when data runs outside the current view.
    /// 0 for a view that slides every time new data is added
    /// 1 for a view that only slides forward when new data runs off the screen
    /// </summary>
    public double PaddingFractionX { get; set; } = 0;

    /// <summary>
    /// Defines the amount of whitespace added to the top or bottom of the data when data runs outside the current view.
    /// 0 sets axis limits to tightly fit the data height
    /// 1 sets axis limits to double the vertical span in the direction of the vertical overflow
    /// </summary>
    public double PaddingFractionY { get; set; } = .5;

    public CoordinateRange GetRangeX(CoordinateRange viewRangeX, CoordinateRange dataRangeX)
    {
        double padHorizontal = Width * PaddingFractionX;

        bool   xOverflow = dataRangeX.Max > viewRangeX.Max || dataRangeX.Max < viewRangeX.Min;
        double xMax      = xOverflow ? dataRangeX.Max + padHorizontal : viewRangeX.Max;
        double xMin      = xOverflow ? xMax - Width : viewRangeX.Min;

        return new CoordinateRange(xMin, xMax);
    }

    public CoordinateRange GetRangeY(CoordinateRange viewRangeY, CoordinateRange dataRangeY)
    {
        double yMin = dataRangeY.Min - dataRangeY.Min * PaddingFractionY;
        double yMax = dataRangeY.Max + dataRangeY.Max * PaddingFractionY;

        return new CoordinateRange(yMin, yMax);
    }
}
