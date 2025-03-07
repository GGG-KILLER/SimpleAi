using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using MIConvexHull;
using ScottPlot;
using ScottPlot.Plottables;
using SimpleAi.UI.Maths;

namespace SimpleAi.UI.ViewModels;

internal sealed partial class FruitTrainingViewModel : TrainingBaseViewModel
{
    private Polygon? _safeAreaPolygon;

    public FruitTrainingViewModel() : base(LossFunction.BinaryCrossEntropy)
    {
        HiddenActivationFunction = ActivationFunction.ReLu;
        OutputActivationFunction = ActivationFunction.Softmax;
        LearningRate             = 0.05f;
        LearningRateDecay        = 0.04f;
        BatchSize                = 20;
        HiddenLayers             = "10";
        UseMultiThreading        = true;
    }

    [ObservableProperty]
    public partial Vector2DRange TotalArea { get; set; } = new(new Vector2D(X: 0, Y: 0), new Vector2D(X: 9, Y: 14));

    [ObservableProperty]
    public partial Vector2DRange SafeArea { get; set; } = new(new Vector2D(X: 0, Y: 0), new Vector2D(X: 5.5f, Y: 8));

    [ObservableProperty]
    public partial int SafePoints { get; set; } = 37;

    [ObservableProperty]
    public partial int UnsafePoints { get; set; } = 87;

    public Plot? TrainingDataPlot { get; set; }

    /// <inheritdoc />
    protected override int NetworkInputs => 2;

    /// <inheritdoc />
    protected override int NetworkOutputs => 2;

    /// <inheritdoc />
    protected override ValueTask<ITrainingData<NumberTypeT>> GetTrainingData()
    {
        TrainingDataPlot!.Clear();
        return GetTestData();
    }

    /// <inheritdoc />
    protected override ValueTask<ITrainingData<NumberTypeT>> GetTestData()
    {
        TrainingDataPoint<NumberTypeT>[] trainingData = TrainingHelpers.GenerateTrainingData(
            TotalArea,
            SafeArea,
            UnsafePoints,
            SafePoints);

        Coordinates[] safeDataPoints = trainingData
                                       .Where(static x => Tensor.IndexOfMax<NumberTypeT>(x.ExpectedOutputs) == 0)
                                       .Select(static x => new Coordinates(x.Inputs[0], x.Inputs[1])).ToArray();
        Coordinates[] unsafeDataPoints = trainingData
                                         .Where(static x => Tensor.IndexOfMax<NumberTypeT>(x.ExpectedOutputs) == 1)
                                         .Select(static x => new Coordinates(x.Inputs[0], x.Inputs[1])).ToArray();
        TrainingDataPlot!.Add.ScatterPoints(safeDataPoints, Colors.Green);
        TrainingDataPlot.Add.ScatterPoints(unsafeDataPoints, Colors.Red);
        TrainingDataPlot.Axes.AutoScale();
        Refresh!();

        return ValueTask.FromResult<ITrainingData<NumberTypeT>>(new InMemoryTrainingData<NumberTypeT>(trainingData));
    }

    /// <inheritdoc />
    protected override void OnPlotsSetup()
    {
        IList<Vector2D> safeArea = ConvexHull();

        _safeAreaPolygon = TrainingDataPlot!.Add.Polygon(
            safeArea.Select(static vec => new Coordinates(vec.X, vec.Y)).ToArray());
        _safeAreaPolygon.LineWidth = 0;
        _safeAreaPolygon.FillColor = Colors.LightGray.WithAlpha(alpha: .5);
    }

    /// <inheritdoc />
    protected override void OnPostTrainingUpdate()
    {
        var safeArea = ConvexHull();
        _safeAreaPolygon!.UpdateCoordinates(safeArea.Select(static vec => new Coordinates(vec.X, vec.Y)).ToArray());
    }

    private IList<Vector2D> ConvexHull()
    {
        const NumberTypeT delta    = 0.25f;
        var               vertexes = new List<Vector2D>();

        for (NumberTypeT x = TotalArea.Start.X; x <= TotalArea.End.X; x += delta)
        {
            for (NumberTypeT y = TotalArea.Start.Y; y <= TotalArea.End.Y; y += delta)
            {
                var results = NeuralNetwork!.RunInference((NumberTypeT[]) [x, y]);
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
}
