using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Platform.Storage;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using MIConvexHull;
using MsBox.Avalonia;
using MsBox.Avalonia.Base;
using MsBox.Avalonia.Enums;
using ScottPlot;
using ScottPlot.Plottables;
using SimpleAi.UI.IO;
using SimpleAi.UI.Maths;
using SimpleAi.UI.Plotting;

namespace SimpleAi.UI.ViewModels;

internal sealed partial class FruitTrainingViewModel : ObservableObject
{
    private static readonly char[] s_hiddenLayersSplitters = [',', ';', ':'];

    private readonly SemaphoreSlim                 _neuralNetworkSemaphore = new(initialCount: 1, maxCount: 1);
    private          INetworkTrainer<NumberTypeT>? _networkTrainer;
    private          NeuralNetwork<NumberTypeT>?   _neuralNetwork;

    [ObservableProperty, NotifyCanExecuteChangedFor(nameof(StartTrainingCommand))]
    public partial bool IsTraining { get; private set; }

    [ObservableProperty]
    public partial Vector2DRange TotalArea { get; set; } = new(new Vector2D(X: 0, Y: 0), new Vector2D(X: 9, Y: 14));

    [ObservableProperty]
    public partial Vector2DRange SafeArea { get; set; } = new(new Vector2D(X: 0, Y: 0), new Vector2D(X: 5.5f, Y: 8));

    [ObservableProperty]
    public partial ActivationFunction HiddenActivationFunction { get; set; } = ActivationFunction.ReLu;

    [ObservableProperty]
    public partial ActivationFunction OutputActivationFunction { get; set; } = ActivationFunction.SoftMax;

    [ObservableProperty]
    public partial CostFunction CostFunction { get; set; } = CostFunction.CrossEntropy;

    [ObservableProperty]
    public partial NumberTypeT LearningRate { get; set; } = 0.05f;

    [ObservableProperty]
    public partial NumberTypeT LearningRateDecay { get; set; } = 0.04f;

    [ObservableProperty]
    public partial int BatchSize { get; set; } = 20;

    // ReSharper disable once RedundantDefaultMemberInitializer (It's better to be explicit in this case)
    [ObservableProperty]
    public partial NumberTypeT WeightsMean { get; set; } = 0;

    [ObservableProperty]
    public partial NumberTypeT WeightsStdDev { get; set; } = 1;

    [ObservableProperty]
    public partial int SafePoints { get; set; } = 37;

    [ObservableProperty]
    public partial int UnsafePoints { get; set; } = 87;

    [ObservableProperty]
    public partial string HiddenLayers { get; set; } = "10";

    [ObservableProperty]
    public partial bool UseMultiThreading { get; set; } = false;

    public Plot? TrainingDataPlot { get; set; }

    public Plot? CostPlot { get; set; }

    public Plot? LearningRatePlot { get; set; }

    public Plot? AccuracyPlot { get; set; }

    public Action? Refresh { get; set; }

    private bool CanExecuteStartTraining => !IsTraining;

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
                static x => x.ExpectedOutputs[0] == 1 && x.ExpectedOutputs[1] == 0).Select(
                static x => new Coordinates(x.Inputs[0], x.Inputs[1])).ToArray();
            Coordinates[] unsafeDataPoints = trainingData.Where(
                // ReSharper disable once CompareOfFloatsByEqualityOperator
                static x => x.ExpectedOutputs[0] == 0 && x.ExpectedOutputs[1] == 1).Select(
                static x => new Coordinates(x.Inputs[0], x.Inputs[1])).ToArray();
            TrainingDataPlot.Add.ScatterPoints(safeDataPoints, Colors.Green);
            TrainingDataPlot.Add.ScatterPoints(unsafeDataPoints, Colors.Red);
            TrainingDataPlot.Axes.AutoScale();
            Refresh!();

            int[] layerSizes = HiddenLayers.Split(
                                               s_hiddenLayersSplitters,
                                               StringSplitOptions.RemoveEmptyEntries
                                               | StringSplitOptions.RemoveEmptyEntries).Select(int.Parse)
                                           .ToArray();

            var layers = new List<Layer<NumberTypeT>>(layerSizes.Length + 1);
            for (var idx = 0; idx < layerSizes.Length; idx++)
            {
                int inputs  = idx == 0 ? 2 : layerSizes[idx - 1];
                int outputs = layerSizes[idx];
                layers.Add(
                    HiddenActivationFunction switch
                    {
                        ActivationFunction.Sigmoid => new Layer<NumberTypeT, Sigmoid<NumberTypeT>>(inputs, outputs),
                        ActivationFunction.TanH => new Layer<NumberTypeT, TanH<NumberTypeT>>(inputs, outputs),
                        ActivationFunction.ReLu => new Layer<NumberTypeT, ReLu<NumberTypeT>>(inputs, outputs),
                        ActivationFunction.SiLu => new Layer<NumberTypeT, SiLu<NumberTypeT>>(inputs, outputs),
                        ActivationFunction.SoftMax => new Layer<NumberTypeT, SoftMax<NumberTypeT>>(inputs, outputs),
                        _ => throw new InvalidOperationException(message: "Invalid activation function."),
                    });
            }

            // Create output layer
            {
                int outputLayerInputs = layerSizes.Length > 0 ? layerSizes[^1] : 2;
                layers.Add(
                    OutputActivationFunction switch
                    {
                        ActivationFunction.Sigmoid =>
                            new Layer<NumberTypeT, Sigmoid<NumberTypeT>>(outputLayerInputs, neurons: 2),
                        ActivationFunction.TanH =>
                            new Layer<NumberTypeT, TanH<NumberTypeT>>(outputLayerInputs, neurons: 2),
                        ActivationFunction.ReLu =>
                            new Layer<NumberTypeT, ReLu<NumberTypeT>>(outputLayerInputs, neurons: 2),
                        ActivationFunction.SiLu =>
                            new Layer<NumberTypeT, SiLu<NumberTypeT>>(outputLayerInputs, neurons: 2),
                        ActivationFunction.SoftMax => new Layer<NumberTypeT, SoftMax<NumberTypeT>>(
                            outputLayerInputs,
                            neurons: 2),
                        _ => throw new InvalidOperationException(message: "Invalid activation function."),
                    });
            }

            _neuralNetwork = new NeuralNetwork<NumberTypeT>(layers.ToArray());

            _networkTrainer = CostFunction switch
            {
                CostFunction.MeanSquaredError => new NetworkTrainer<NumberTypeT, MeanSquaredError<NumberTypeT>>(
                    _neuralNetwork,
                    trainingData,
                    new TrainingParameters<NumberTypeT>
                    {
                        InitialLearnRate    = LearningRate,
                        LearnRateDecay      = LearningRateDecay,
                        BatchSize           = BatchSize > 0 ? BatchSize : null,
                        ParallelizeTraining = UseMultiThreading,
                    }),
                CostFunction.CrossEntropy => new NetworkTrainer<NumberTypeT, CrossEntropy<NumberTypeT>>(
                    _neuralNetwork,
                    trainingData,
                    new TrainingParameters<NumberTypeT>
                    {
                        InitialLearnRate    = LearningRate,
                        LearnRateDecay      = LearningRateDecay,
                        BatchSize           = BatchSize > 0 ? BatchSize : null,
                        ParallelizeTraining = UseMultiThreading,
                    }),
                _ => throw new InvalidOperationException(message: "Invalid cost function."),
            };

            CostPlot!.Clear();
            LearningRatePlot!.Clear();
            AccuracyPlot!.Clear();

            IList<Vector2D> safeArea = ConvexHull();
            Polygon safeAreaPolygon = TrainingDataPlot.Add.Polygon(
                safeArea.Select(static vec => new Coordinates(vec.X, vec.Y)).ToArray());
            safeAreaPolygon.LineWidth = 0;
            safeAreaPolygon.FillColor = Colors.LightGray.WithAlpha(alpha: .5);

            DataLogger costPlot = CostPlot.Add.DataLogger();
            costPlot.Axes.XAxis            = CostPlot.Axes.Bottom;
            costPlot.Axes.XAxis.Label.Text = "Epoch";
            costPlot.LegendText            = "Cost (should go down ideally)";
            costPlot.ManageAxisLimits      = true;
            costPlot.AxisManager           = new ConstantSlide { PaddingFractionY = .01, Width = 100 };
            costPlot.Add(_networkTrainer.Epoch, _networkTrainer.CalculateAverageCost());

            DataLogger learningRatePlot = LearningRatePlot.Add.DataLogger();
            learningRatePlot.Axes.XAxis            = LearningRatePlot.Axes.Bottom;
            learningRatePlot.Axes.XAxis.Label.Text = "Epoch";
            learningRatePlot.LegendText            = "Learning rate";
            learningRatePlot.ManageAxisLimits      = true;
            learningRatePlot.AxisManager           = new ConstantSlide { PaddingFractionY = .01, Width = 100 };
            learningRatePlot.Add(_networkTrainer.Epoch, _networkTrainer.LearningRate);

            DataLogger accuracyPlot = AccuracyPlot.Add.DataLogger();
            accuracyPlot.Axes.XAxis            = AccuracyPlot.Axes.Bottom;
            accuracyPlot.Axes.XAxis.Label.Text = "Epoch";
            accuracyPlot.LegendText            = "Accuracy (should go up ideally)";
            accuracyPlot.ManageAxisLimits      = true;
            accuracyPlot.AxisManager           = new ConstantSlide { PaddingFractionY = .01, Width = 100 };
            accuracyPlot.Add(_networkTrainer.Epoch, _neuralNetwork.CalculateAccuracy(trainingData));

            Refresh!();

            var sw = Stopwatch.StartNew();
            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    await _neuralNetworkSemaphore.WaitAsync(cancellationToken)
                                                 .ConfigureAwait(continueOnCapturedContext: false);
                    _networkTrainer.RunTrainingIteration();
                }
                finally
                {
                    _neuralNetworkSemaphore.Release();
                }
                costPlot.Add(_networkTrainer.Epoch, _networkTrainer.CalculateAverageCost());
                learningRatePlot.Add(_networkTrainer.Epoch, _networkTrainer.LearningRate);
                accuracyPlot.Add(_networkTrainer.Epoch, _neuralNetwork.CalculateAccuracy(trainingData));

                if (sw.ElapsedMilliseconds >= 100)
                {
                    sw.Restart();

                    safeArea = ConvexHull();
                    safeAreaPolygon.UpdateCoordinates(
                        safeArea.Select(static vec => new Coordinates(vec.X, vec.Y)).ToArray());
                }

                Refresh!();
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            Console.WriteLine(ex);
            throw;
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

        for (NumberTypeT x = TotalArea.Start.X; x <= TotalArea.End.X; x += delta)
        {
            for (NumberTypeT y = TotalArea.Start.Y; y <= TotalArea.End.Y; y += delta)
            {
                var results = _neuralNetwork!.RunInference((NumberTypeT[]) [x, y]);
                if (TrainingHelpers.IsSafeish(results)) vertexes.Add(new Vector2D(x, y));
            }
        }

        if (vertexes.Count < 2) return [];

        ConvexHullCreationResult<Vector2D> result = MIConvexHull.ConvexHull.Create2D(vertexes, delta);
        if (result.Outcome == ConvexHullCreationResultOutcome.Success) return result.Result;

        Console.WriteLine(
            $"Error generating convex hull: {result.Outcome} | {result.ErrorMessage} (points: {string.Join(separator: ", ", vertexes)})");
        return [];
    }

    [RelayCommand(AllowConcurrentExecutions = false, FlowExceptionsToTaskScheduler = true)]
    private async Task SaveModelAsync(Window window, CancellationToken cancellationToken = default)
    {
        try
        {
            if (_neuralNetwork is null)
            {
                IMsBox<ButtonResult> msgBox = MessageBoxManager.GetMessageBoxStandard(
                    title: "No Network Under Training!",
                    text: "Please start training a network to save it.",
                    ButtonEnum.Ok,
                    Icon.Error);
                await msgBox.ShowAsPopupAsync(window).ConfigureAwait(continueOnCapturedContext: false);
                return;
            }
            using IStorageFile? file = await window.StorageProvider.SaveFilePickerAsync(
                                           new FilePickerSaveOptions
                                           {
                                               Title            = "Save AI Model",
                                               DefaultExtension = ".sai.nn",
                                               FileTypeChoices =
                                               [
                                                   new FilePickerFileType(name: "Neural Network Model")
                                                   {
                                                       Patterns = ["*.sai.nn"],
                                                   },
                                               ],
                                               ShowOverwritePrompt = true,
                                           }).ConfigureAwait(continueOnCapturedContext: false);
            if (file is null)
            {
                IMsBox<ButtonResult> msgBox = MessageBoxManager.GetMessageBoxStandard(
                    title: "No File Selected!",
                    text: "Please select a file to save your model.",
                    ButtonEnum.Ok,
                    Icon.Error);
                await msgBox.ShowAsPopupAsync(window).ConfigureAwait(continueOnCapturedContext: false);
                return;
            }

            await using (Stream stream = await file.OpenWriteAsync().ConfigureAwait(continueOnCapturedContext: false))
            {
                try
                {
                    await _neuralNetworkSemaphore.WaitAsync(cancellationToken)
                                                 .ConfigureAwait(continueOnCapturedContext: false);
                    await ModelSerializer.SaveModelAsync(_neuralNetwork, stream)
                                         .ConfigureAwait(continueOnCapturedContext: false);
                }
                finally
                {
                    _neuralNetworkSemaphore.Release();
                }
            }

            {
                IMsBox<ButtonResult> msgBox = MessageBoxManager.GetMessageBoxStandard(
                    title: "Model Saved!",
                    $"Model successfully saved to {file.Name}.",
                    ButtonEnum.Ok,
                    Icon.Success);
                await msgBox.ShowAsPopupAsync(window).ConfigureAwait(continueOnCapturedContext: false);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine(ex);
            throw;
        }
    }
}
