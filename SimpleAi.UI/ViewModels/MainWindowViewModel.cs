using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using ScottPlot;
using ScottPlot.Plottables;

namespace SimpleAi.UI.ViewModels;

internal sealed partial class MainWindowViewModel : ObservableObject
{
    private static readonly char[]                        s_hiddenLayersSplitters = [',', ';', ':'];
    private                 INeuralNetwork<NumberTypeT>?  _neuralNetwork;
    private                 TrainingSession<NumberTypeT>? _trainingSession;

    [ObservableProperty, NotifyCanExecuteChangedFor(nameof(StartTrainingCommand))]
    public partial bool IsTraining { get; private set; }

    [ObservableProperty, NotifyCanExecuteChangedFor(nameof(StartTrainingCommand))]
    public partial bool IsCommandInProgress { get; private set; }

    [ObservableProperty]
    public partial (VectorTypeT Start, VectorTypeT End) TotalArea { get; set; } = ((0, 0), (100, 100));

    [ObservableProperty]
    public partial (VectorTypeT Start, VectorTypeT End) SafeArea { get; set; } = ((0, 0), (20, 20));

    [ObservableProperty]
    public partial ActivationFunction ActivationFunction { get; set; } = ActivationFunction.Sigmoid;

    [ObservableProperty]
    public partial CostFunction CostFunction { get; set; } = CostFunction.MeanSquaredError;

    [ObservableProperty]
    public partial NumberTypeT LearningRate { get; set; } = 0.05;

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
            await Task.Delay(millisecondsDelay: 1, cancellationToken).ConfigureAwait(continueOnCapturedContext: false);

            TrainingDataPoint<NumberTypeT>[] trainingData = GenerateTrainingData();
            TrainingDataPoint<NumberTypeT>[] checkData    = GenerateTrainingData();

            TrainingDataPlot!.Clear();
            Coordinates[] safeDataPoints = trainingData.Concat(checkData).Where(
                                                           // ReSharper disable once CompareOfFloatsByEqualityOperator
                                                           x => x.ExpectedOutputs.Span[index: 0] == 1
                                                                && x.ExpectedOutputs.Span[index: 1] == 0)
                                                       .Select(
                                                           x => new Coordinates(
                                                               x.Inputs.Span[index: 0],
                                                               x.Inputs.Span[index: 1])).ToArray();
            Coordinates[] unsafeDataPoints = trainingData.Concat(checkData).Where(
                                                             // ReSharper disable once CompareOfFloatsByEqualityOperator
                                                             x => x.ExpectedOutputs.Span[index: 0] == 0
                                                                  && x.ExpectedOutputs.Span[index: 1] == 1)
                                                         .Select(
                                                             x => new Coordinates(
                                                                 x.Inputs.Span[index: 0],
                                                                 x.Inputs.Span[index: 1])).ToArray();
            Scatter safePointsPlottable   = TrainingDataPlot.Add.ScatterPoints(safeDataPoints, Colors.Green);
            Scatter unsafePointsPlottable = TrainingDataPlot.Add.ScatterPoints(unsafeDataPoints, Colors.Red);
            TrainingDataPlot.Axes.AutoScale([safePointsPlottable, unsafePointsPlottable]);
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
            var        iterations   = 0;
            DataLogger costPlot     = CostPlot.Add.DataLogger();
            DataLogger accuracyPlot = AccuracyPlot.Add.DataLogger();
            costPlot.ViewJump(paddingFraction: 0);
            accuracyPlot.ViewJump(paddingFraction: 0);
            costPlot.Add(iterations, _neuralNetwork.AverageCost(_trainingSession));
            accuracyPlot.Add(iterations, CalculateAccuracy(checkData));
            costPlot.Axes.XAxis                = CostPlot.Axes.Bottom;
            costPlot.Axes.XAxis.Label.Text     = "Generation";
            costPlot.LegendText                = "Cost (should go down ideally)";
            accuracyPlot.Axes.XAxis            = AccuracyPlot.Axes.Bottom;
            accuracyPlot.Axes.XAxis.Label.Text = "Generation";
            accuracyPlot.LegendText            = "Accuracy (should go up ideally)";
            Refresh!();

            while (!cancellationToken.IsCancellationRequested)
            {
                _trainingSession.ShuffleTrainingData();
                _neuralNetwork.Train(_trainingSession, LearningRate);
                iterations++;
                costPlot.Add(iterations, _neuralNetwork.AverageCost(_trainingSession));
                accuracyPlot.Add(iterations, CalculateAccuracy(checkData));
                Refresh!();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine(ex);
        }
        finally
        {
            IsTraining = false;
        }

        return;

        TrainingDataPoint<NumberTypeT>[] GenerateTrainingData()
        {
            var trainingData        = new TrainingDataPoint<NumberTypeT>[SafePoints + UnsafePoints];
            var trainingDataInputs  = new NumberTypeT[(SafePoints + UnsafePoints) * 2];
            var trainingDataOutputs = new NumberTypeT[(SafePoints + UnsafePoints) * 2];
            var generatedSafe       = 0;
            var generatedUnsafe     = 0;

            for (var idx = 0; idx < trainingData.Length; idx++)
            {
                bool isSafe;
                if (generatedSafe < SafePoints && generatedUnsafe < UnsafePoints)
                    isSafe = Random.Shared.NextSingle() > 0.5f;
                else
                    isSafe = generatedSafe < SafePoints;

                if (isSafe)
                {
                    trainingDataInputs[(idx * 2) + 0] = SafeArea.Start.X
                                                        + (Random.Shared.NextSingle()
                                                           * (SafeArea.End.X - SafeArea.Start.X));
                    trainingDataInputs[(idx * 2) + 1] = SafeArea.Start.Y
                                                        + (Random.Shared.NextSingle()
                                                           * (SafeArea.End.Y - SafeArea.Start.Y));
                    trainingDataOutputs[(idx * 2) + 0] = 1;
                    trainingDataOutputs[(idx * 2) + 1] = 0;
                    generatedSafe++;
                }
                else
                {
                regen:
                    trainingDataInputs[(idx * 2) + 0] = TotalArea.Start.X
                                                        + (Random.Shared.NextSingle()
                                                           * (TotalArea.End.X - TotalArea.Start.X));
                    trainingDataInputs[(idx * 2) + 1] = TotalArea.Start.Y
                                                        + (Random.Shared.NextSingle()
                                                           * (TotalArea.End.Y - TotalArea.Start.Y));

                    if (SafeArea.Start.X < trainingDataInputs[(idx * 2) + 0]
                        && trainingDataInputs[(idx * 2) + 0] < SafeArea.End.X
                        && SafeArea.Start.Y < trainingDataInputs[(idx * 2) + 1]
                        && trainingDataInputs[(idx * 2) + 1] < SafeArea.End.Y)
                        goto regen;

                    trainingDataOutputs[(idx * 2) + 0] = 0;
                    trainingDataOutputs[(idx * 2) + 1] = 1;
                    generatedUnsafe++;
                }

                trainingData[idx] = new TrainingDataPoint<NumberTypeT>(
                    trainingDataInputs.AsMemory(idx * 2, length: 2),
                    trainingDataOutputs.AsMemory(idx * 2, length: 2));
            }

            return trainingData;
        }

        double CalculateAccuracy(TrainingDataPoint<NumberTypeT>[] trainingDataPoints)
        {
            var               hits   = 0;
            Span<NumberTypeT> output = stackalloc NumberTypeT[2];

            foreach (TrainingDataPoint<NumberTypeT> point in trainingDataPoints)
            {
                _neuralNetwork!.RunInference(_trainingSession!.InferenceSession, point.Inputs.Span, output);
                // ReSharper disable once CompareOfFloatsByEqualityOperator (These comparisons don't have that risk)
                if (point.ExpectedOutputs.Span[index: 0] == 1 && point.ExpectedOutputs.Span[index: 1] == 0)
                {
                    if (output[index: 0] > output[index: 1]) hits++;
                }
                else
                {
                    if (output[index: 0] < output[index: 1]) hits++;
                }
            }

            return (double) hits / trainingDataPoints.Length;
        }
    }
}
