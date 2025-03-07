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
using MsBox.Avalonia;
using MsBox.Avalonia.Enums;
using ScottPlot;
using ScottPlot.AxisLimitManagers;
using ScottPlot.Plottables;
using SimpleAi.UI.IO;

namespace SimpleAi.UI.ViewModels;

internal abstract partial class TrainingBaseViewModel(LossFunction lossFunction) : ObservableObject
{
    private static readonly char[]                        s_hiddenLayersSplitters = [',', ';', ':'];
    protected readonly      SemaphoreSlim                 NeuralNetworkSemaphore  = new(initialCount: 1, maxCount: 1);
    protected               NeuralNetwork<NumberTypeT>?   NeuralNetwork;
    protected               INetworkTrainer<NumberTypeT>? NetworkTrainer;

    [ObservableProperty, NotifyCanExecuteChangedFor(nameof(StartTrainingCommand))]
    protected partial bool IsTraining { get; private set; }

    [ObservableProperty]
    public partial ActivationFunction HiddenActivationFunction { get; set; }

    [ObservableProperty]
    public partial ActivationFunction OutputActivationFunction { get; set; }

    [ObservableProperty]
    public partial NumberTypeT LearningRate { get; set; }

    [ObservableProperty]
    public partial NumberTypeT LearningRateDecay { get; set; }

    [ObservableProperty]
    public partial int BatchSize { get; set; }

    [ObservableProperty]
    public partial string HiddenLayers { get; set; } = "";

    [ObservableProperty]
    public partial bool UseMultiThreading { get; set; }

    public Plot? CostPlot { get; set; }

    public Plot? LearningRatePlot { get; set; }

    public Plot? AccuracyPlot { get; set; }

    public Action? Refresh { get; set; }

    protected virtual bool CanExecuteStartTraining => !IsTraining;

    protected abstract int NetworkInputs { get; }

    protected abstract int NetworkOutputs { get; }

    [RelayCommand(
        CanExecute = nameof(CanExecuteStartTraining),
        AllowConcurrentExecutions = false,
        IncludeCancelCommand = true,
        FlowExceptionsToTaskScheduler = true)]
    protected virtual async Task OnStartTrainingAsync(CancellationToken cancellationToken)
    {
        IsTraining = true;
        try
        {
            // Leave UI Thread (hopefully)
            await Task.Delay(millisecondsDelay: 1, cancellationToken).ConfigureAwait(continueOnCapturedContext: false);

            ITrainingData<NumberTypeT> trainingData = await GetTrainingData();
            ITrainingData<NumberTypeT> testData     = await GetTestData();

            int[] layerSizes = HiddenLayers.Split(
                                               s_hiddenLayersSplitters,
                                               StringSplitOptions.RemoveEmptyEntries
                                               | StringSplitOptions.RemoveEmptyEntries).Select(int.Parse)
                                           .ToArray();

            var layers = new List<Layer<NumberTypeT>>(layerSizes.Length + 1);
            for (var idx = 0; idx < layerSizes.Length; idx++)
            {
                int inputs  = idx == 0 ? NetworkInputs : layerSizes[idx - 1];
                int outputs = layerSizes[idx];
                layers.Add(CreateLayer(HiddenActivationFunction, inputs, outputs));
            }

            // Create output layer
            int outputLayerInputs = layerSizes.Length > 0 ? layerSizes[^1] : NetworkInputs;
            layers.Add(CreateLayer(OutputActivationFunction, outputLayerInputs, NetworkOutputs));

            var trainingParameters = new TrainingParameters<NumberTypeT>
            {
                InitialLearnRate    = LearningRate,
                LearnRateDecay      = LearningRateDecay,
                BatchSize           = BatchSize > 0 ? BatchSize : null,
                ParallelizeTraining = UseMultiThreading,
            };
            NeuralNetwork = new NeuralNetwork<NumberTypeT>(layers.ToArray());
            NetworkTrainer = lossFunction switch
            {
                LossFunction.MeanSquaredError =>
                    new NetworkTrainer<NumberTypeT, MeanSquaredError<NumberTypeT>>(
                        NeuralNetwork,
                        trainingData,
                        trainingParameters),
                LossFunction.BinaryCrossEntropy => new NetworkTrainer<NumberTypeT, BinaryCrossEntropy<NumberTypeT>>(
                    NeuralNetwork,
                    trainingData,
                    trainingParameters),
                LossFunction.MultiClassCrossEntropy =>
                    new NetworkTrainer<NumberTypeT, MultiClassCrossEntropy<NumberTypeT>>(
                        NeuralNetwork,
                        trainingData,
                        trainingParameters),
                _ => throw new InvalidOperationException(message: "Invalid cost function."),
            };

            CostPlot!.Clear();
            LearningRatePlot!.Clear();
            AccuracyPlot!.Clear();

            OnPlotsSetup();

            DataLogger costPlot = CostPlot.Add.DataLogger();
            costPlot.Axes.XAxis            = CostPlot.Axes.Bottom;
            costPlot.Axes.XAxis.Label.Text = "Epoch";
            costPlot.LegendText            = "Cost (should go down ideally)";
            costPlot.ManageAxisLimits      = true;
            costPlot.AxisManager           = new Slide { PaddingFractionY = .01, Width = 100 };
            if (ShouldDoPostTrainingUpdate(0))
                costPlot.Add(NetworkTrainer.Epoch, NetworkTrainer.CalculateAverageLoss());

            DataLogger learningRatePlot = LearningRatePlot.Add.DataLogger();
            learningRatePlot.Axes.XAxis            = LearningRatePlot.Axes.Bottom;
            learningRatePlot.Axes.XAxis.Label.Text = "Epoch";
            learningRatePlot.LegendText            = "Learning rate";
            learningRatePlot.ManageAxisLimits      = true;
            learningRatePlot.AxisManager           = new Slide { PaddingFractionY = .01, Width = 100 };
            learningRatePlot.Add(NetworkTrainer.Epoch, NetworkTrainer.LearningRate);

            DataLogger accuracyPlot = AccuracyPlot.Add.DataLogger();
            accuracyPlot.Axes.XAxis            = AccuracyPlot.Axes.Bottom;
            accuracyPlot.Axes.XAxis.Label.Text = "Epoch";
            accuracyPlot.LegendText            = "Accuracy (should go up ideally)";
            accuracyPlot.ManageAxisLimits      = true;
            accuracyPlot.AxisManager           = new Slide { PaddingFractionY = .01, Width = 100 };
            if (ShouldDoPostTrainingUpdate(0))
                accuracyPlot.Add(NetworkTrainer.Epoch, NeuralNetwork.CalculateAccuracy(testData));

            Refresh!();

            OnTrainingStart();

            var sw = Stopwatch.StartNew();
            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    await NeuralNetworkSemaphore.WaitAsync(cancellationToken)
                                                .ConfigureAwait(continueOnCapturedContext: false);

                    OnPreTraining();
                    NetworkTrainer.RunTrainingIteration();
                    OnPostTraining();
                    learningRatePlot.Add(NetworkTrainer.Epoch, NetworkTrainer.LearningRate);

                    if (ShouldDoPostTrainingUpdate(sw.ElapsedMilliseconds))
                    {
                        sw.Restart();
                        OnPostTrainingUpdate();
                        costPlot.Add(NetworkTrainer.Epoch, NetworkTrainer.CalculateAverageLoss());
                        accuracyPlot.Add(NetworkTrainer.Epoch, NeuralNetwork.CalculateAccuracy(testData));
                    }
                }
                finally
                {
                    NeuralNetworkSemaphore.Release();
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
            try { OnTrainingEnd(); }
            finally { IsTraining = false; }
        }
    }

    private static Layer<NumberTypeT> CreateLayer(ActivationFunction activationFunction, int inputs, int neurons)
    {
        return activationFunction switch
        {
            ActivationFunction.Sigmoid => new Layer<NumberTypeT, Sigmoid<NumberTypeT>>(inputs, neurons),
            ActivationFunction.TanH    => new Layer<NumberTypeT, TanH<NumberTypeT>>(inputs, neurons),
            ActivationFunction.ReLu    => new Layer<NumberTypeT, ReLu<NumberTypeT>>(inputs, neurons),
            ActivationFunction.SiLu    => new Layer<NumberTypeT, SiLu<NumberTypeT>>(inputs, neurons),
            ActivationFunction.Softmax => new Layer<NumberTypeT, Softmax<NumberTypeT>>(inputs, neurons),
            _                          => throw new InvalidOperationException(message: "Invalid activation function."),
        };
    }

    protected abstract ValueTask<ITrainingData<NumberTypeT>> GetTrainingData();

    protected abstract ValueTask<ITrainingData<NumberTypeT>> GetTestData();

    protected virtual void OnPlotsSetup() { }

    protected virtual void OnTrainingStart() { }

    protected virtual void OnPreTraining() { }

    protected virtual void OnPostTraining() { }

    protected virtual bool ShouldDoPostTrainingUpdate(long elapsedMilliseconds) => elapsedMilliseconds >= 100;

    protected virtual void OnPostTrainingUpdate() { }

    protected virtual void OnTrainingEnd() { }

    [RelayCommand(AllowConcurrentExecutions = false, FlowExceptionsToTaskScheduler = true)]
    private async Task SaveModelAsync(Window window, CancellationToken cancellationToken = default)
    {
        try
        {
            if (NeuralNetwork is null)
            {
                await MessageBoxManager
                      .GetMessageBoxStandard(
                          "No Network Under Training!",
                          "Please start training a network to save it.",
                          ButtonEnum.Ok,
                          Icon.Error).ShowAsPopupAsync(window).ConfigureAwait(continueOnCapturedContext: false);
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
                await MessageBoxManager
                      .GetMessageBoxStandard(
                          "No File Selected!",
                          "Please select a file to save your model.",
                          ButtonEnum.Ok,
                          Icon.Error).ShowAsPopupAsync(window).ConfigureAwait(continueOnCapturedContext: false);
                return;
            }

            await using (Stream stream = await file.OpenWriteAsync().ConfigureAwait(continueOnCapturedContext: false))
            {
                try
                {
                    await NeuralNetworkSemaphore.WaitAsync(cancellationToken)
                                                .ConfigureAwait(continueOnCapturedContext: false);
                    await ModelSerializer.SaveModelAsync(NeuralNetwork, stream)
                                         .ConfigureAwait(continueOnCapturedContext: false);
                }
                finally
                {
                    NeuralNetworkSemaphore.Release();
                }
            }

            await MessageBoxManager
                  .GetMessageBoxStandard(
                      "Model Saved!",
                      $"Model successfully saved to {file.Name}.",
                      ButtonEnum.Ok,
                      Icon.Success).ShowAsPopupAsync(window).ConfigureAwait(continueOnCapturedContext: false);
        }
        catch (Exception ex)
        {
            Console.WriteLine(ex);
            throw;
        }
    }
}
