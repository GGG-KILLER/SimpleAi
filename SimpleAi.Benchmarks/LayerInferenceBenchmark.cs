using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;

namespace SimpleAi.Benchmarks;

[SimpleJob(RunStrategy.Throughput)]
[MemoryRandomization]
public class LayerInferenceBenchmark
{
    private Layer<float> _floatLayer;
    private Layer<double> _doubleLayer;
    private float[]? _floatInputs;
    private double[]? _doubleInputs;

    [Params(5, 10, 250, 5000, 10_000)]
    public int Inputs { get; set; }

    [Params(5, 10, 250, 5000, 10_000)]
    public int Neurons { get; set; }

    [GlobalSetup]
    public void GlobalSetup()
    {
        _floatLayer = new Layer<float>(Inputs, Neurons);
        _floatLayer.Randomize(Random.Shared.Next(1, 101));
        _doubleLayer = new Layer<double>(Inputs, Neurons);
        _doubleLayer.Randomize(Random.Shared.Next(1, 101));

        _floatInputs = new float[Inputs];
        for (var idx = 0; idx < _floatInputs.Length; idx++)
            _floatInputs[idx] = Random.Shared.NextSingle() * Random.Shared.Next();

        _doubleInputs = new double[Inputs];
        for (var idx = 0; idx < _doubleInputs.Length; idx++)
            _doubleInputs[idx] = Random.Shared.NextDouble() * Random.Shared.Next();
    }

    [Benchmark]
    public float[] FloatInfer()
    {
        var output = GC.AllocateUninitializedArray<float>(Neurons);
        _floatLayer.RunInference(_floatInputs, output);
        return output;
    }

    [Benchmark]
    public double[] DoubleInfer()
    {
        var output = GC.AllocateUninitializedArray<double>(Neurons);
        _doubleLayer.RunInference(_doubleInputs, output);
        return output;
    }
}
