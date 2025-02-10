using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;

namespace SimpleAi.Benchmarks;

[SimpleJob(RunStrategy.Throughput)]
[MemoryRandomization]
public class LayerInferenceBenchmark
{
    private Layer<float, ReLU<float>>? _floatLayer;
    private Layer<double, ReLU<double>>? _doubleLayer;
    private float[]? _floatInputs;
    private double[]? _doubleInputs;
    private float[]? _floatOutputs;
    private double[]? _doubleOutputs;

    [Params(5, 10, 250, 5000, 10_000)]
    public int Inputs { get; set; }

    [Params(5, 10, 250, 5000, 10_000)]
    public int Neurons { get; set; }

    [GlobalSetup]
    public void GlobalSetup()
    {
        _floatLayer = new Layer<float, ReLU<float>>(Inputs, Neurons);
        _floatLayer.Randomize(Random.Shared.Next(1, 101));
        _doubleLayer = new Layer<double, ReLU<double>>(Inputs, Neurons);
        _doubleLayer.Randomize(Random.Shared.Next(1, 101));

        _floatInputs = new float[Inputs];
        for (var idx = 0; idx < _floatInputs.Length; idx++)
            _floatInputs[idx] = Random.Shared.NextSingle() * Random.Shared.Next();
        _floatOutputs = new float[Neurons];

        _doubleInputs = new double[Inputs];
        for (var idx = 0; idx < _doubleInputs.Length; idx++)
            _doubleInputs[idx] = Random.Shared.NextDouble() * Random.Shared.Next();
        _doubleOutputs = new double[Neurons];
    }

    [Benchmark]
    public void FloatInfer() => _floatLayer!.RunInference(_floatInputs, _floatOutputs);

    [Benchmark]
    public void DoubleInfer() => _doubleLayer!.RunInference(_doubleInputs, _doubleOutputs);
}
