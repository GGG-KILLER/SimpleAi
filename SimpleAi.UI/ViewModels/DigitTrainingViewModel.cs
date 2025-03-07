using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using MsBox.Avalonia;
using MsBox.Avalonia.Enums;
using Parquet;
using Parquet.Schema;
using SkiaSharp;

namespace SimpleAi.UI.ViewModels;

internal sealed partial class DigitTrainingViewModel : TrainingBaseViewModel
{
    private DigitTrainingData? _digitTrainingData;
    private DigitTrainingData? _digitTestingData;
    private int                _currentImageIndex;

    public DigitTrainingViewModel() : base(LossFunction.MultiClassCrossEntropy)
    {
        HiddenActivationFunction = ActivationFunction.ReLu;
        OutputActivationFunction = ActivationFunction.Softmax;
        LearningRate             = 0.05;
        LearningRateDecay        = 0.04;
        BatchSize                = 32;
        HiddenLayers             = "300";
        UseMultiThreading        = true;
    }

    [ObservableProperty, NotifyPropertyChangedFor(nameof(TrainingDataFileName))]
    [SuppressMessage("ReSharper", "MemberCanBeMadeStatic.Local", Justification = "False positive.")]
    private partial IStorageFile? TrainingDataFile { get; set; }

    public string TrainingDataFileName => TrainingDataFile?.Name ?? "Select a file...";

    [ObservableProperty, NotifyPropertyChangedFor(nameof(TestingDataFileName))]
    [SuppressMessage("ReSharper", "MemberCanBeMadeStatic.Local", Justification = "False positive.")]
    private partial IStorageFile? TestingDataFile { get; set; }

    public string TestingDataFileName => TestingDataFile?.Name ?? "Select a file...";

    [ObservableProperty]
    public partial bool DoGraphUpdates { get; set; } = true;

    [ObservableProperty]
    public partial IImage? CurrentImage { get; set; }

    [ObservableProperty]
    public partial string CurrentImageClassification { get; set; } = "";

    [ObservableProperty]
    public partial string StatusText { get; set; } = "Select a training file.";

    [RelayCommand(AllowConcurrentExecutions = false, FlowExceptionsToTaskScheduler = true)]
    private async Task SelectTrainingDataFileAsync(Window window)
    {
        var file = await window.StorageProvider.OpenFilePickerAsync(
                       new FilePickerOpenOptions
                       {
                           Title         = "Select a training data file",
                           AllowMultiple = false,
                           FileTypeFilter =
                           [
                               new FilePickerFileType(name: "Parquet file") { Patterns = ["*.parquet"] }
                           ],
                       }).ConfigureAwait(false);
        if (file.Count < 1) return;

        TrainingDataFile?.Dispose();
        TrainingDataFile = file[0];
        _digitTrainingData?.Dispose();
        _digitTrainingData = null;
        StatusText         = TestingDataFile is null ? "Select a testing data file." : "Ready to start training.";
    }

    [RelayCommand(AllowConcurrentExecutions = false, FlowExceptionsToTaskScheduler = true)]
    private async Task SelectTestingDataFileAsync(Window window)
    {
        var file = await window.StorageProvider.OpenFilePickerAsync(
                       new FilePickerOpenOptions
                       {
                           Title         = "Select a testing data file",
                           AllowMultiple = false,
                           FileTypeFilter =
                           [
                               new FilePickerFileType(name: "Parquet file") { Patterns = ["*.parquet"] }
                           ],
                       }).ConfigureAwait(false);
        if (file.Count < 1) return;

        TestingDataFile?.Dispose();
        TestingDataFile = file[0];
        _digitTestingData?.Dispose();
        _digitTestingData = null;
    }

    [RelayCommand(AllowConcurrentExecutions = false, FlowExceptionsToTaskScheduler = true)]
    private async Task GoToPreviousImageAsync()
    {
        await NeuralNetworkSemaphore.WaitAsync().ConfigureAwait(false);
        try
        {
            // We also want this to behave as a ring ("overflow" back to length - 1) on subtraction at zero.
            if (_currentImageIndex == 0)
                _currentImageIndex = _digitTestingData!.Length - 1;
            else
                _currentImageIndex--;
            UpdateCurrentImage();
        }
        finally
        {
            NeuralNetworkSemaphore.Release();
        }
    }

    [RelayCommand(AllowConcurrentExecutions = false, FlowExceptionsToTaskScheduler = true)]
    private async Task GoToRandomImageAsync()
    {
        await NeuralNetworkSemaphore.WaitAsync().ConfigureAwait(false);
        try
        {
            _currentImageIndex = Random.Shared.Next(0, _digitTestingData!.Length);
            UpdateCurrentImage();
        }
        finally
        {
            NeuralNetworkSemaphore.Release();
        }
    }

    [RelayCommand(AllowConcurrentExecutions = false, FlowExceptionsToTaskScheduler = true)]
    private async Task GoToNextImageAsync()
    {
        await NeuralNetworkSemaphore.WaitAsync().ConfigureAwait(false);
        try
        {
            _currentImageIndex = (_currentImageIndex + 1) % _digitTestingData!.Length;
            UpdateCurrentImage();
        }
        finally
        {
            NeuralNetworkSemaphore.Release();
        }
    }

    private void UpdateCurrentImage()
    {
        (Bitmap? bitmap, long expected, TrainingDataPoint<NumberTypeT> trainingDataPoint) =
            _digitTestingData![_currentImageIndex];
        CurrentImage = bitmap;
        Tensor<NumberTypeT> result           = NeuralNetwork!.RunInference(trainingDataPoint.Inputs);
        int                 actual           = Tensor.IndexOfMax<NumberTypeT>(result);
        NumberTypeT         actualConfidence = result[actual];
        CurrentImageClassification =
            $"Classification: {actual} (confidence: {actualConfidence:P}, expected: {expected})";
    }

    /// <inheritdoc />
    protected override async ValueTask<ITrainingData<NumberTypeT>> GetTrainingData()
    {
        StatusText = "Loading training data...";
        if (TrainingDataFile is null)
        {
            await MessageBoxManager.GetMessageBoxStandard(
                title: "No training data!",
                text: """
                      No training data file was selected.
                      Please select a training data file.
                      """,
                ButtonEnum.Ok,
                Icon.Error).ShowAsync().ConfigureAwait(false);
            StatusText = "No training data file was selected.";
            throw new InvalidOperationException("No training data file selected.");
        }

        if (_digitTrainingData is not null) return _digitTrainingData;
        var sw = Stopwatch.StartNew();
        {
            await using var stream = await TrainingDataFile.OpenReadAsync().ConfigureAwait(false);
            _digitTrainingData = await DigitTrainingData.LoadAsync(stream).ConfigureAwait(false);
        }
        sw.Stop();
        Console.WriteLine($"Training data loaded in {sw.Elapsed.TotalMilliseconds}ms");
        StatusText = "Training data loaded.";
        return _digitTrainingData;
    }

    /// <inheritdoc />
    protected override async ValueTask<ITrainingData<NumberTypeT>> GetTestData()
    {
        StatusText = "Loading test data...";
        if (TestingDataFile is null)
        {
            await MessageBoxManager.GetMessageBoxStandard(
                title: "No testing data!",
                text: """
                      No testing data file was selected.
                      Please select a training data file.
                      """,
                ButtonEnum.Ok,
                Icon.Error).ShowAsync().ConfigureAwait(false);
            StatusText = "No testing data file was selected.";
            throw new InvalidOperationException("No testing data file selected.");
        }

        if (_digitTestingData is not null) return _digitTestingData;
        var sw = Stopwatch.StartNew();
        {
            await using var stream = await TestingDataFile.OpenReadAsync().ConfigureAwait(false);
            _digitTestingData = await DigitTrainingData.LoadAsync(stream).ConfigureAwait(false);
        }
        sw.Stop();
        Console.WriteLine($"Test data loaded in {sw.Elapsed.TotalMilliseconds}ms");
        StatusText = "Test data loaded.";
        return _digitTestingData;
    }

    /// <inheritdoc />
    protected override int NetworkInputs => _digitTrainingData![0].DataPoint.Inputs.Length;

    /// <inheritdoc />
    protected override int NetworkOutputs => _digitTrainingData![0].DataPoint.ExpectedOutputs.Length;

    /// <inheritdoc />
    protected override void OnPlotsSetup()
    {
        StatusText = "Setting up plots...";

        _currentImageIndex = 0;
        UpdateCurrentImage();
    }

    private readonly Stopwatch _trainingSw = new Stopwatch();

    /// <inheritdoc />
    protected override void OnTrainingStart()
    {
        StatusText = "Starting training...";
    }

    /// <inheritdoc />
    protected override void OnPreTraining()
    {
        StatusText = $"Training epoch {NetworkTrainer!.Epoch + ((double) BatchSize / _digitTrainingData!.Length)}...";
        _trainingSw.Restart();
    }

    /// <inheritdoc />
    protected override void OnPostTraining()
    {
        _trainingSw.Stop();
        Console.WriteLine($"Epoch {NetworkTrainer!.Epoch}: {_trainingSw.Elapsed.TotalMilliseconds}ms");
        StatusText = $"Done training epoch {NetworkTrainer!.Epoch}.";
    }

    /// <inheritdoc />
    protected override bool ShouldDoPostTrainingUpdate(long elapsedMilliseconds)
        => DoGraphUpdates && NetworkTrainer!.Epoch % 1 == 0;

    /// <inheritdoc />
    protected override void OnPostTrainingUpdate()
    {
        StatusText = "Updating plots...";
    }

    /// <inheritdoc />
    protected override void OnTrainingEnd()
    {
        StatusText = "Training finished.";
    }
}

internal sealed class DigitTrainingData : ITrainingData<NumberTypeT>, IDisposable
{
    private readonly TrainingRecord[] _records;

    private DigitTrainingData(IEnumerable<TrainingRecord> records)
    {
        _records = [..records];
    }

    /// <inheritdoc />
    public int Length => _records.Length;

    public TrainingRecord this[Index index] => _records[index];

    /// <inheritdoc />
    TrainingDataPoint<NumberTypeT> ITrainingData<NumberTypeT>.this[Index index] => _records[index].DataPoint;

    /// <inheritdoc />
    ReadOnlyMemory<TrainingDataPoint<NumberTypeT>> ITrainingData<NumberTypeT>.this[Range range]
        => Array.ConvertAll(_records[range], static x => x.DataPoint);

    /// <inheritdoc />
    public IEnumerator<TrainingDataPoint<NumberTypeT>> GetEnumerator()
    {
        return _records.Select(static record => record.DataPoint).GetEnumerator();
    }

    /// <inheritdoc />
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <inheritdoc />
    public void Shuffle() => Random.Shared.Shuffle(_records);

    /// <inheritdoc />
    public void Dispose()
    {
        for (var index = 0; index < _records.Length; index++)
        {
            TrainingRecord record = _records[index];
            _records[index] = default(TrainingRecord);
            record.Bitmap?.Dispose();
        }
    }

    public static async Task<DigitTrainingData> LoadAsync(Stream stream)
    {
        using var reader          = await ParquetReader.CreateAsync(stream).ConfigureAwait(false);
        var       records         = new List<TrainingRecord>((int) reader.RowGroups.Sum(static x => x.RowCount));
        var       imageBytesField = reader.Schema.FindDataField(new FieldPath(["image", "bytes"]));
        var       imageLabelField = reader.Schema.FindDataField("label");
        for (var rowGroup = 0; rowGroup < reader.RowGroupCount; rowGroup++)
        {
            using var rowGroupReader   = reader.OpenRowGroupReader(rowGroup);
            var       imageBytesColumn = await rowGroupReader.ReadColumnAsync(imageBytesField).ConfigureAwait(false);
            var       imageLabelColumn = await rowGroupReader.ReadColumnAsync(imageLabelField).ConfigureAwait(false);
            var       imageBytesSpan   = imageBytesColumn.AsSpan<byte[]>();
            var       imageLabelSpan   = imageLabelColumn.AsSpan<long?>();
            for (var row = 0; row < imageBytesSpan.Length; row++)
            {
                var           bitmap = new Bitmap(new MemoryStream(imageBytesSpan[row]));
                long          label  = imageLabelSpan[row]!.Value;
                NumberTypeT[] input  = ConvertImage(imageBytesSpan[row]);
                var           output = new NumberTypeT[10];
                output[label] = 1;

                records.Add(new TrainingRecord(bitmap, label, new TrainingDataPoint<NumberTypeT>(input, output)));
            }
        }

        return new DigitTrainingData(records);

        static NumberTypeT[] ConvertImage(byte[] rawImage)
        {
            using var bitmap = SKBitmap.Decode(rawImage);
            return Array.ConvertAll(
                bitmap.Pixels,
                // Images are already in grayscale
                static color => color.Green / (NumberTypeT) 255);
        }
    }

    public readonly record struct TrainingRecord(Bitmap Bitmap, long Label, TrainingDataPoint<NumberTypeT> DataPoint);
}
