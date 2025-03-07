using Avalonia.Controls;
using ScottPlot;
using ScottPlot.MultiplotLayouts;
using SimpleAi.UI.ViewModels;

namespace SimpleAi.UI.Views;

internal sealed partial class DigitTraining : UserControl
{
    public DigitTraining()
    {
        InitializeComponent();

        IMultiplot multiPlot = Plot.Multiplot;
        multiPlot.AddPlots(total: 3);

        var vm = (DigitTrainingViewModel) DataContext!;
        vm.CostPlot         = multiPlot.GetPlot(index: 0);
        vm.LearningRatePlot = multiPlot.GetPlot(index: 1);
        vm.AccuracyPlot     = multiPlot.GetPlot(index: 2);
        vm.Refresh          = Plot.Refresh;

        vm.CostPlot.Title(text: "Cost");
        vm.LearningRatePlot.Title(text: "Learning Rate");
        vm.AccuracyPlot.Title(text: "Accuracy");

        const int rows    = 1;
        const int columns = 3;
        var       layout  = new CustomGrid();
        layout.Set(vm.CostPlot, new GridCell(rowIndex: 0, colIndex: 0, rows, columns));
        layout.Set(vm.LearningRatePlot, new GridCell(rowIndex: 0, colIndex: 1, rows, columns));
        layout.Set(vm.AccuracyPlot, new GridCell(rowIndex: 0, colIndex: 2, rows, columns));
        multiPlot.Layout = layout;
    }
}
