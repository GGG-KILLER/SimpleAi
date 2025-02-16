using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using JetBrains.Annotations;

namespace SimpleAi.Benchmarks;

[UsedImplicitly(ImplicitUseKindFlags.Access, ImplicitUseTargetFlags.WithMembers)]
[SimpleJob(RunStrategy.Throughput), MemoryRandomization]
public class LayerInferenceBenchmark
{
    private double[]?                    _doubleInputs;
    private Layer<double, ReLU<double>>? _doubleLayer;
    private double[]?                    _doubleOutputs;
    private float[]?                     _floatInputs;
    private Layer<float, ReLU<float>>?   _floatLayer;
    private float[]?                     _floatOutputs;

    [Params(5, 10, 250, 5000, 10_000)]
    public int Inputs { get; [UsedImplicitly] set; }

    [Params(5, 10, 250, 5000, 10_000)]
    public int Neurons { get; [UsedImplicitly] set; }

    [GlobalSetup]
    public void GlobalSetup()
    {
        _floatLayer = new Layer<float, ReLU<float>>(Inputs, Neurons);
        _floatLayer.RandomizeWeights(mean: 0, stdDev: 1);
        _doubleLayer = new Layer<double, ReLU<double>>(Inputs, Neurons);
        _doubleLayer.RandomizeWeights(mean: 0, stdDev: 1);

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
