using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using MIConvexHull;
using ScottPlot;
using ScottPlot.AxisLimitManagers;
using ScottPlot.Plottables;
using SimpleAi.UI.Maths;

namespace SimpleAi.UI.ViewModels;

internal sealed partial class MainWindowViewModel : ObservableObject
{
    private static readonly char[] s_hiddenLayersSplitters = [',', ';', ':'];

    private INeuralNetwork<NumberTypeT>?  _neuralNetwork;
    private TrainingSession<NumberTypeT>? _trainingSession;

    [ObservableProperty, NotifyCanExecuteChangedFor(nameof(StartTrainingCommand))]
    public partial bool IsTraining { get; private set; }

    [ObservableProperty, NotifyCanExecuteChangedFor(nameof(StartTrainingCommand))]
    public partial bool IsCommandInProgress { get; private set; }

    [ObservableProperty]
    public partial Vector2DRange TotalArea { get; set; } = new(new Vector2D(0, 0), new Vector2D(100, 100));

    [ObservableProperty]
    public partial Vector2DRange SafeArea { get; set; } = new(new Vector2D(0, 0), new Vector2D(20, 20));

    [ObservableProperty]
    public partial ActivationFunction ActivationFunction { get; set; } = ActivationFunction.Sigmoid;

    [ObservableProperty]
    public partial CostFunction CostFunction { get; set; } = CostFunction.NaiveSquaredError;

    [ObservableProperty]
    public partial NumberTypeT LearningRate { get; set; } = 0.05f;

    // ReSharper disable once RedundantDefaultMemberInitializer (It's better to be explicit in this case)
    [ObservableProperty]
    public partial NumberTypeT WeightsMean { get; set; } = 0;

    [ObservableProperty]
    public partial NumberTypeT WeightsStdDev { get; set; } = 1;

    [ObservableProperty]
    public partial int SafePoints { get; set; } = 40;

    [ObservableProperty]
    public partial int UnsafePoints { get; set; } = 60;

    [ObservableProperty]
    public partial string HiddenLayers { get; set; } = string.Empty;

    public Plot? TrainingDataPlot { get; set; }

    public Plot? CostPlot { get; set; }

    public Plot? AccuracyPlot { get; set; }

    public Action? Refresh { get; set; }

    private bool CanExecuteStartTraining => !IsTraining && !IsCommandInProgress;

    [RelayCommand(
        CanExecute = nameof(CanExecuteStartTraining),
        AllowConcurrentExecutions = false,
        IncludeCancelCommand = true,
        FlowExceptionsToTaskScheduler = true)]
    private async Task OnStartTrainingAsync(CancellationToken cancellationToken)
    {
        IsTraining = true;
        try
        {
            // Leave UI Thread (hopefully)
            await Task.Delay(millisecondsDelay: 1, cancellationToken).ConfigureAwait(continueOnCapturedContext: false);

            TrainingDataPoint<NumberTypeT>[] trainingData = TrainingHelpers.GenerateTrainingData(
                TotalArea,
                SafeArea,
                UnsafePoints,
                SafePoints);
            // TrainingDataPoint<NumberTypeT>[] checkData    = GenerateTrainingData();

            TrainingDataPlot!.Clear();
            Coordinates[] safeDataPoints = trainingData.Where(
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                static x => x.ExpectedOutputs.Span[index: 0] == 1 && x.ExpectedOutputs.Span[index: 1] == 0).Select(
                static x => new Coordinates(x.Inputs.Span[index: 0], x.Inputs.Span[index: 1])).ToArray();
            Coordinates[] unsafeDataPoints = trainingData.Where(
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                static x => x.ExpectedOutputs.Span[index: 0] == 0 && x.ExpectedOutputs.Span[index: 1] == 1).Select(
                static x => new Coordinates(x.Inputs.Span[index: 0], x.Inputs.Span[index: 1])).ToArray();
            TrainingDataPlot.Add.ScatterPoints(safeDataPoints, Colors.Green);
            TrainingDataPlot.Add.ScatterPoints(unsafeDataPoints, Colors.Red);
            TrainingDataPlot.Axes.SetLimits(
                TotalArea.Start.X - 5,
                TotalArea.End.X + 5,
                TotalArea.Start.Y - 5,
                TotalArea.End.Y + 5);
            Refresh!();

            int[] layers = HiddenLayers.Split(
                                           s_hiddenLayersSplitters,
                                           StringSplitOptions.RemoveEmptyEntries
                                           | StringSplitOptions.RemoveEmptyEntries).Select(int.Parse)
                                       .Append(element: 2).ToArray();
            _neuralNetwork = ActivationFunction switch
            {
                ActivationFunction.ReLU => new NeuralNetwork<NumberTypeT, ReLU<NumberTypeT>>(inputs: 2, layers),
                ActivationFunction.Sigmoid =>
                    new NeuralNetwork<NumberTypeT, Sigmoid<NumberTypeT>>(inputs: 2, layers),
                ActivationFunction.TanH => new NeuralNetwork<NumberTypeT, TanH<NumberTypeT>>(inputs: 2, layers),
                ActivationFunction.SoftMax => new NeuralNetwork<NumberTypeT, SoftMax<NumberTypeT>>(
                    inputs: 2,
                    layers),
                _ => throw new InvalidOperationException(message: "Invalid activation function."),
            };
            _trainingSession = CostFunction switch
            {
                CostFunction.NaiveSquaredError =>
                    new TrainingSession<NumberTypeT, NaiveSquaredError<NumberTypeT>>(trainingData, _neuralNetwork),
                CostFunction.MeanSquaredError =>
                    new TrainingSession<NumberTypeT, MeanSquaredError<NumberTypeT>>(trainingData, _neuralNetwork),
                CostFunction.CrossEntropy => new TrainingSession<NumberTypeT, CrossEntropy<NumberTypeT>>(
                    trainingData,
                    _neuralNetwork),
                _ => throw new InvalidOperationException(message: "Invalid cost function."),
            };

            _neuralNetwork.RandomizeWeights(WeightsMean, WeightsStdDev);

            CostPlot!.Clear();
            AccuracyPlot!.Clear();
            var iterations = 0;

            var safeArea = ConvexHull();
            Polygon safeAreaPolygon = TrainingDataPlot.Add.Polygon(
                safeArea.Select(static vec => new Coordinates(vec.X, vec.Y)).ToArray());
            safeAreaPolygon.LineWidth = 0;
            safeAreaPolygon.FillColor = Colors.LightGray.WithAlpha(.5);

            DataLogger costPlot = CostPlot.Add.DataLogger();
            costPlot.Axes.XAxis            = CostPlot.Axes.Bottom;
            costPlot.Axes.XAxis.Label.Text = "Generation";
            costPlot.LegendText            = "Cost (should go down ideally)";
            costPlot.ManageAxisLimits      = true;
            costPlot.AxisManager           = new Slide { PaddingFractionX = 0, PaddingFractionY = 0.25, Width = 100 };
            costPlot.Add(iterations, _neuralNetwork.AverageCost(_trainingSession));

            DataLogger accuracyPlot = AccuracyPlot.Add.DataLogger();
            accuracyPlot.Axes.XAxis = AccuracyPlot.Axes.Bottom;
            accuracyPlot.Axes.XAxis.Label.Text = "Generation";
            accuracyPlot.LegendText = "Accuracy (should go up ideally)";
            accuracyPlot.ManageAxisLimits = true;
            accuracyPlot.AxisManager = new Slide { PaddingFractionX = 0, PaddingFractionY = 0.25, Width = 100 };
            accuracyPlot.Add(
                iterations,
                _neuralNetwork.CalculateAccuracy(_trainingSession.InferenceSession, trainingData));

            Refresh!();

            while (!cancellationToken.IsCancellationRequested)
            {
                _trainingSession.ShuffleTrainingData();
                _neuralNetwork.Train(_trainingSession, LearningRate);
                iterations++;
                costPlot.Add(iterations, _neuralNetwork.AverageCost(_trainingSession));
                accuracyPlot.Add(
                    iterations,
                    _neuralNetwork.CalculateAccuracy(_trainingSession.InferenceSession, trainingData));

                if (iterations % 100 == 0)
                {
                    safeArea = ConvexHull();
                    safeAreaPolygon.UpdateCoordinates(
                        safeArea.Select(static vec => new Coordinates(vec.X, vec.Y)).ToArray());
                }
                Refresh!();
            }
        }
        finally
        {
            IsTraining = false;
        }
    }

    private IList<Vector2D> ConvexHull()
    {
        const NumberTypeT delta    = 0.25f;
        var               vertexes = new List<Vector2D>();

        Span<NumberTypeT> results = stackalloc NumberTypeT[2];
        for (NumberTypeT x = TotalArea.Start.X; x <= TotalArea.End.X; x += delta)
        {
            for (NumberTypeT y = TotalArea.Start.Y; y <= TotalArea.End.Y; y += delta)
            {
                results.Clear();
                _neuralNetwork!.RunInference(_trainingSession!.InferenceSession, [x, y], results);
                if (TrainingHelpers.IsSafeish(results)) vertexes.Add(new Vector2D(x, y));
            }
        }

        if (vertexes.Count < 2) return [];

        ConvexHullCreationResult<Vector2D> result = MIConvexHull.ConvexHull.Create2D(vertexes, tolerance: delta);
        if (result.Outcome == ConvexHullCreationResultOutcome.Success) return result.Result;

        Console.WriteLine(
            $"Error generating convex hull: {result.Outcome} | {result.ErrorMessage} (points: {string.Join(", ", vertexes)})");
        return [];
    }
}
