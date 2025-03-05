using System.Numerics.Tensors;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Environments;
using BenchmarkDotNet.Jobs;
using JetBrains.Annotations;

namespace SimpleAi.Benchmarks;

[UsedImplicitly(ImplicitUseKindFlags.Access, ImplicitUseTargetFlags.WithMembers), Config(typeof(Config))]
public class LayerInferenceBenchmark
{
    private double[]?                    _doubleInputs;
    private Layer<double, ReLu<double>>? _doubleLayer;
    private float[]?                     _floatInputs;
    private Layer<float, ReLu<float>>?   _floatLayer;

    [Params(5, 10, 250, 5000, 10_000)]
    public int Inputs { get; [UsedImplicitly] set; }

    [Params(5, 10, 250, 5000, 10_000)]
    public int Neurons { get; [UsedImplicitly] set; }

    [GlobalSetup]
    public void GlobalSetup()
    {
        _floatLayer  = new Layer<float, ReLu<float>>(Inputs, Neurons);
        _doubleLayer = new Layer<double, ReLu<double>>(Inputs, Neurons);

        _floatInputs = new float[Inputs];
        for (var idx = 0; idx < _floatInputs.Length; idx++)
            _floatInputs[idx] = Random.Shared.NextSingle() * Random.Shared.Next();

        _doubleInputs = new double[Inputs];
        for (var idx = 0; idx < _doubleInputs.Length; idx++)
            _doubleInputs[idx] = Random.Shared.NextDouble() * Random.Shared.Next();
    }

    [Benchmark]
    public Tensor<float> FloatInfer() => _floatLayer!.RunInference(_floatInputs);

    [Benchmark]
    public Tensor<double> DoubleInfer() => _doubleLayer!.RunInference(_doubleInputs);

    private sealed class Config : ManualConfig
    {
        public Config()
        {
            var job = Job.Default.WithArguments([new MsBuildArgument("/p:NoWarn=SYSLIB5001")])
                         .WithStrategy(RunStrategy.Throughput).WithMemoryRandomization();

            AddDiagnoser(MemoryDiagnoser.Default);

            AddJob(job.WithRuntime(CoreRuntime.Core90));
        }
    }
}
