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
        multiPlot.AddPlots(3);

        var vm = (MainWindowViewModel) DataContext!;
        vm.TrainingDataPlot = multiPlot.GetPlot(0);
        vm.CostPlot         = multiPlot.GetPlot(1);
        vm.AccuracyPlot     = multiPlot.GetPlot(2);
        vm.Refresh          = Plot.Refresh;

        vm.TrainingDataPlot.Title("Training Data");
        vm.CostPlot.Title("Cost");
        vm.AccuracyPlot.Title("Accuracy");

        var layout = new CustomGrid();
        layout.Set(multiPlot.GetPlot(0), new GridCell(0, 0, 4, 2, 2, 2));
        layout.Set(multiPlot.GetPlot(1), new GridCell(1, 0, 2, 2));
        layout.Set(multiPlot.GetPlot(2), new GridCell(1, 1, 2, 2));
        multiPlot.Layout = layout;
    }
}
