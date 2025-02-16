using Avalonia.Controls;
using ScottPlot;
using ScottPlot.MultiplotLayouts;
using SimpleAi.UI.ViewModels;

namespace SimpleAi.UI.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();

        IMultiplot multiPlot = Plot.Multiplot;
        multiPlot.AddPlots(total: 3);

        var vm = (MainWindowViewModel) DataContext!;
        vm.TrainingDataPlot = multiPlot.GetPlot(index: 0);
        vm.CostPlot         = multiPlot.GetPlot(index: 1);
        vm.AccuracyPlot     = multiPlot.GetPlot(index: 2);
        vm.Refresh          = Plot.Refresh;

        vm.TrainingDataPlot.Title(text: "Training Data");
        vm.CostPlot.Title(text: "Cost");
        vm.AccuracyPlot.Title(text: "Accuracy");

        var layout = new CustomGrid();
        layout.Set(
            vm.TrainingDataPlot,
            new GridCell(rowIndex: 0, colIndex: 0, rowCount: 4, colCount: 2, rowSpan: 2, colSpan: 2));
        layout.Set(vm.CostPlot, new GridCell(rowIndex: 1, colIndex: 0, rowCount: 2, colCount: 2));
        layout.Set(vm.AccuracyPlot, new GridCell(rowIndex: 1, colIndex: 1, rowCount: 2, colCount: 2));
        multiPlot.Layout = layout;
    }
}
