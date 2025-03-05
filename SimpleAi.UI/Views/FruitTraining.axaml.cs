using Avalonia.Controls;
using ScottPlot;
using ScottPlot.MultiplotLayouts;
using SimpleAi.UI.ViewModels;

namespace SimpleAi.UI.Views;

internal sealed partial class FruitTraining : UserControl
{
    public FruitTraining()
    {
        InitializeComponent();

        IMultiplot multiPlot = Plot.Multiplot;
        multiPlot.AddPlots(total: 4);

        var vm = (FruitTrainingViewModel) DataContext!;
        vm.TrainingDataPlot = multiPlot.GetPlot(index: 0);
        vm.CostPlot         = multiPlot.GetPlot(index: 1);
        vm.LearningRatePlot = multiPlot.GetPlot(index: 2);
        vm.AccuracyPlot     = multiPlot.GetPlot(index: 3);
        vm.Refresh          = Plot.Refresh;

        vm.TrainingDataPlot.Title(text: "Training Data");
        vm.CostPlot.Title(text: "Cost");
        vm.LearningRatePlot.Title(text: "Learning Rate");
        vm.AccuracyPlot.Title(text: "Accuracy");

        const int rows    = 2;
        const int columns = 3;
        var       layout  = new CustomGrid();
        // TODO: Fix this when ScottPlot has a fixed version (colSpan and rowSpan are switched around).
        layout.Set(vm.TrainingDataPlot, new GridCell(rowIndex: 0, colIndex: 0, rows, columns, columns));
        layout.Set(vm.CostPlot, new GridCell(rowIndex: 1, colIndex: 0, rows, columns));
        layout.Set(vm.LearningRatePlot, new GridCell(rowIndex: 1, colIndex: 1, rows, columns));
        layout.Set(vm.AccuracyPlot, new GridCell(rowIndex: 1, colIndex: 2, rows, columns));
        multiPlot.Layout = layout;
    }
}
