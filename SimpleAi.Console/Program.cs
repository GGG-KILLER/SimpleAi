// See https://aka.ms/new-console-template for more information

using System.ComponentModel.DataAnnotations;
using System.Globalization;
using System.Numerics;
using System.Text.RegularExpressions;
using Cocona;
using SimpleAi;
using Spectre.Console;

CoconaLiteApp.Run((
    [Argument][RegularExpression(Regexes.RangePattern)] string safeRangeStr,
    [Argument][RegularExpression(Regexes.RangePattern)] string totalRangeStr,
    [Argument] float learnRate,
    [Option][RegularExpression(@"^(|\d+([,:;]\d+)*)$")] string hiddenLayers = "",
    [Option] float weightsMean = 0,
    [Option] float weightsStdDev = 1,
    [Option] int safePointsCount = 20,
    [Option] int unsafePointsCount = 90,
    [Option] ActivationFunction activationFunction = ActivationFunction.Sigmoid,
    [Option] CostFunction costFunction = CostFunction.MeanSquaredError) =>
{
    var stderr = AnsiConsole.Create(new AnsiConsoleSettings { Out = new AnsiConsoleOutput(Console.Error) });

    var safeRange = parseRange(safeRangeStr);
    var totalRange = parseRange(totalRangeStr);

    AnsiConsole.MarkupLineInterpolated($"Safe range: [green]{safeRange.Start}[/] to [green]{safeRange.End}[/]");
    AnsiConsole.MarkupLineInterpolated($"Total range: [green]{totalRange.Start}[/] to [green]{totalRange.End}[/]");

    if (safeRange.Start.X < totalRange.Start.X || safeRange.Start.Y < totalRange.Start.Y || safeRange.End.X > totalRange.End.X || safeRange.End.Y > totalRange.End.Y)
    {
        stderr.MarkupLineInterpolated($"[red]error:[/] safe range is not contained within the total range.");
        return -1;
    }

    var trainingData = new TrainingDataPoint<float>[safePointsCount + unsafePointsCount];
    var checkData = new TrainingDataPoint<float>[safePointsCount + unsafePointsCount];

    generateTrainingData(
        safePointsCount,
        unsafePointsCount,
        safeRange,
        totalRange,
        trainingData,
        out var minX,
        out var minY,
        out var maxX,
        out var maxY);

    generateTrainingData(
        safePointsCount,
        unsafePointsCount,
        safeRange,
        totalRange,
        checkData,
        out minX,
        out minY,
        out maxX,
        out maxY);

    drawTrainingData(totalRange, [.. trainingData, .. checkData], minX, minY, maxX, maxY);

    var layers = hiddenLayers.Split([',', ':', ';', ' '], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
        .Select(int.Parse)
        .Append(2)
        .ToArray();
    INeuralNetwork<float> neuralNetwork = activationFunction switch
    {
        ActivationFunction.ReLU => new NeuralNetwork<float, ReLU<float>>(2, layers),
        ActivationFunction.Sigmoid => new NeuralNetwork<float, Sigmoid<float>>(2, layers),
        ActivationFunction.SoftMax => new NeuralNetwork<float, SoftMax<float>>(2, layers),
        ActivationFunction.TanH => new NeuralNetwork<float, TanH<float>>(2, layers),
        _ => throw new InvalidOperationException("Invalid activation function."),
    };
    neuralNetwork.RandomizeWeights(weightsMean, weightsStdDev);
    TrainingSession<float> trainingSession = costFunction switch
    {
        CostFunction.MeanSquaredError => new TrainingSession<float, MeanSquaredError<float>>(trainingData, neuralNetwork),
        CostFunction.CrossEntropy => new TrainingSession<float, CrossEntropy<float>>(trainingData, neuralNetwork),
        _ => throw new InvalidOperationException("Invalid cost function."),
    };

    var output = neuralNetwork[^1];
    var cost = neuralNetwork.AverageCost(trainingSession);
    var accuracy = calculateAccuracy(neuralNetwork, trainingSession.InferenceSession, checkData);

    AnsiConsole.MarkupLineInterpolated($"Current state: [[learning rate = {learnRate}, cost = {cost}, accuracy = {accuracy:P}]]");

    while (true)
    {
        var choice = AnsiConsole.Prompt(new TextPrompt<char>("""
            c. Continue
            r. Change learning rate
            q. Quit
            What is your choice?
            """,
            StringComparer.OrdinalIgnoreCase)
            .InvalidChoiceMessage("Invalid choice.")
            .ValidationErrorMessage("Invalid choice.")
            .ShowChoices(true)
            .ShowDefaultValue(true)
            .DefaultValue('c')
            .AddChoice('c')
            .AddChoice('r')
            .AddChoice('q'));

        switch (choice)
        {
            case 'c':
            case 'C':
                neuralNetwork.Train(trainingSession, learnRate);
                cost = neuralNetwork.AverageCost(trainingSession);
                accuracy = calculateAccuracy(neuralNetwork, trainingSession.InferenceSession, checkData);
                AnsiConsole.MarkupLineInterpolated($"Current state: [[learning rate = {learnRate}, cost = {cost}, accuracy = {accuracy:P}]]");
                break;
            case 'r':
            case 'R':
                learnRate = AnsiConsole.Ask<float>($"New learning rate (current is {learnRate}): ");
                goto case 'c';
            case 'q':
            case 'Q':
                goto loopEnd;
        }
    }
loopEnd:;

    return 0;
});


static (Vector2 Start, Vector2 End) parseRange(string input)
{
    var match = Regexes.Range.Match(input);
    return (
        new Vector2(
            float.Parse(match.Groups["startX"].ValueSpan, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture),
            float.Parse(match.Groups["startY"].ValueSpan, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture)),
        new Vector2(
            float.Parse(match.Groups["endX"].ValueSpan, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture),
            float.Parse(match.Groups["endY"].ValueSpan, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture)));
}

static void generateTrainingData(
    int safePointsCount,
    int unsafePointsCount,
    (Vector2 Start, Vector2 End) safeRange,
    (Vector2 Start, Vector2 End) totalRange,
    TrainingDataPoint<float>[] trainingData,
    out float minX,
    out float minY,
    out float maxX,
    out float maxY)
{
    var trainingDataInputs = new float[(safePointsCount + unsafePointsCount) * 2];
    var trainingDataOutputs = new float[(safePointsCount + unsafePointsCount) * 2];
    var checkDataInputs = new float[(safePointsCount + unsafePointsCount) * 2];
    var checkDataOutputs = new float[(safePointsCount + unsafePointsCount) * 2];
    var generatedSafe = 0; var generatedUnsafe = 0;
    minX = float.MaxValue;
    minY = float.MaxValue;
    maxX = float.MinValue;
    maxY = float.MinValue;
    for (var idx = 0; idx < trainingData.Length; idx++)
    {
        var isSafe = false;
        if (generatedSafe < safePointsCount && generatedUnsafe < unsafePointsCount)
            isSafe = Random.Shared.NextSingle() > 0.5f;
        else
            isSafe = generatedSafe < safePointsCount;

        if (isSafe)
        {
            trainingDataInputs[idx * 2 + 0] = safeRange.Start.X + Random.Shared.NextSingle() * (safeRange.End.X - safeRange.Start.X);
            trainingDataInputs[idx * 2 + 1] = safeRange.Start.Y + Random.Shared.NextSingle() * (safeRange.End.Y - safeRange.Start.Y);
            trainingDataOutputs[idx * 2 + 0] = 1;
            trainingDataOutputs[idx * 2 + 1] = 0;
        }
        else
        {
        regen:
            trainingDataInputs[idx * 2 + 0] = totalRange.Start.X + Random.Shared.NextSingle() * (totalRange.End.X - totalRange.Start.X);
            trainingDataInputs[idx * 2 + 1] = totalRange.Start.Y + Random.Shared.NextSingle() * (totalRange.End.Y - totalRange.Start.Y);

            if (safeRange.Start.X < trainingDataInputs[idx * 2 + 0] && trainingDataInputs[idx * 2 + 0] < safeRange.End.X
                && safeRange.Start.Y < trainingDataInputs[idx * 2 + 1] && trainingDataInputs[idx * 2 + 1] < safeRange.End.Y)
                goto regen;

            trainingDataOutputs[idx * 2 + 0] = 0;
            trainingDataOutputs[idx * 2 + 1] = 1;
        }

        minX = float.Min(trainingDataInputs[idx * 2 + 0], minX);
        minY = float.Min(trainingDataInputs[idx * 2 + 1], minY);
        maxX = float.Max(trainingDataInputs[idx * 2 + 0], maxX);
        maxY = float.Max(trainingDataInputs[idx * 2 + 1], maxY);

        trainingData[idx] = new TrainingDataPoint<float>(
            trainingDataInputs.AsMemory(idx * 2, 2),
            trainingDataOutputs.AsMemory(idx * 2, 2));
    }
}

static void drawTrainingData(
    (Vector2 Start, Vector2 End) totalRange,
    IEnumerable<TrainingDataPoint<float>> trainingData,
    float minX,
    float minY,
    float maxX,
    float maxY)
{
    AnsiConsole.Write($"Generated data points: ");
    var first = true;
    foreach (var point in trainingData)
    {
        if (!first)
            AnsiConsole.Write(", ");
        first = false;

        string color;
        if (point.ExpectedOutputs.Span[0] == 1 && point.ExpectedOutputs.Span[1] == 0)
            color = "green";
        else
            color = "red";
        AnsiConsole.Markup($"[{color}]<{point.Inputs.Span[0]}, {point.Inputs.Span[1]}>[/]");
    }
    AnsiConsole.WriteLine();
    AnsiConsole.MarkupLineInterpolated($"Mins and maxes: [green]<{minX}, {minY}>[/] [green]<{maxX}, {maxY}>[/]");

    var canvas = new Canvas(
        AnsiConsole.Profile.Width / 2,
        AnsiConsole.Profile.Height / 2)
    {
        Scale = false,
        PixelWidth = 1
    };
    // We scale by the usable area of the canvas (lastIdx - 1 for 0-based conversion - 2 for borders)
    var scaleX = (canvas.Width - 1 - 2) / (totalRange.End.X - totalRange.Start.X + 1);
    var scaleY = (canvas.Height - 1 - 2) / (totalRange.End.Y - totalRange.Start.Y + 1);
    AnsiConsole.MarkupLineInterpolated($"Canvas size:  [green]<{canvas.Width}, {canvas.Height}>[/]");
    AnsiConsole.MarkupLineInterpolated($"Canvas scale: [green]<{scaleX}, {scaleY}>[/]");

    // Draw borders
    for (var x = 0; x < canvas.Width; x++)
    {
        canvas.SetPixel(x, 0, Color.White);
        canvas.SetPixel(x, canvas.Height - 1, Color.White);
    }
    for (var y = 0; y < canvas.Height; y++)
    {
        canvas.SetPixel(0, y, Color.White);
        canvas.SetPixel(canvas.Width - 1, y, Color.White);
    }
    canvas.SetPixel(1, 1, Color.Blue);
    canvas.SetPixel(canvas.Width - 2, 1, Color.Blue);
    canvas.SetPixel(1, canvas.Height - 2, Color.Blue);
    canvas.SetPixel(canvas.Width - 2, canvas.Height - 2, Color.Blue);

    // Draw points
    foreach (var point in trainingData)
    {
        Color color;
        if (point.ExpectedOutputs.Span[0] == 1 && point.ExpectedOutputs.Span[1] == 0)
            color = Color.Green;
        else
            color = Color.Red;

        canvas.SetPixel(
            1 + (int)Math.Round(Math.Round(point.Inputs.Span[0]) * scaleX),
            canvas.Height - 2 - (int)Math.Round(Math.Round(point.Inputs.Span[1]) * scaleY),
            color);
    }
    AnsiConsole.Write(canvas);
}

static double calculateAccuracy(INeuralNetwork<float> neuralNetwork, InferenceSession<float> inferenceSession, TrainingDataPoint<float>[] trainingDataPoints)
{
    var hits = 0;
    Span<float> output = stackalloc float[2];

    foreach (var point in trainingDataPoints)
    {
        neuralNetwork.RunInference(inferenceSession, point.Inputs.Span, output);
        if (point.ExpectedOutputs.Span[0] == 1 && point.ExpectedOutputs.Span[1] == 0)
        {
            if (output[0] > output[1])
                hits++;
        }
        else
        {
            if (output[0] < output[1])
                hits++;
        }
    }

    return (double)hits / trainingDataPoints.Length;
}

internal static partial class Regexes
{
    public const string RangePattern = @"^\s*(?<startX>\d+(\.\d+)?)\s*,\s*(?<startY>\d+(\.\d+)?):(?<endX>\d+(\.\d+)?)\s*,\s*(?<endY>\d+(\.\d+)?)$";

    [GeneratedRegex(RangePattern, RegexOptions.CultureInvariant | RegexOptions.ExplicitCapture | RegexOptions.IgnoreCase | RegexOptions.Singleline)]
    public static partial Regex Range { get; }
}

internal enum ActivationFunction
{
    Sigmoid,
    TanH,
    ReLU,
    SoftMax,
}

internal enum CostFunction
{
    MeanSquaredError,
    CrossEntropy,
}
