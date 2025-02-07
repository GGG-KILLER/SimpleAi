using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using SimpleAi.Math;

namespace SimpleAi.Benchmarks;

[SimpleJob(RunStrategy.Throughput)]
[IterationTime(250)]
[MemoryRandomization]
public class InferenceImplsBenchmark
{
    private float[]? _weights;
    private float[]? _biases;
    private float[]? _inputs;
    private float[]? _output;

    [Params(5, 10, 250, 5000, 10_000)]
    public int Inputs { get; set; }

    [Params(5, 10, 250, 5000, 10_000)]
    public int Neurons { get; set; }

    [GlobalSetup]
    public void GlobalSetup()
    {
        _inputs = new float[Inputs];
        _weights = new float[Neurons * Inputs];
        _biases = new float[Neurons];
        _output = new float[Neurons];
        for (var idx = 0; idx < _inputs.Length; idx++)
        {
            _inputs[idx] = Random.Shared.NextSingle() * Random.Shared.Next();
        }
        for (var idx = 0; idx < _weights.Length; idx++)
        {
            _weights[idx] = Random.Shared.NextSingle() * Random.Shared.Next();
        }
        for (var idx = 0; idx < _biases.Length; idx++)
        {
            _biases[idx] = Random.Shared.NextSingle() * Random.Shared.Next();
        }
    }

    [Benchmark(Baseline = true)]
    public void CurrentImpl()
    {
        ref float weights = ref _weights!.Ref();

        if (Vector.IsHardwareAccelerated && Vector<float>.IsSupported && _inputs!.Length > Vector<float>.Count * 2)
        {
            var idx = 0;

            Span<Vector<float>> neuronVecAccs = Neurons < 16
                ? stackalloc Vector<float>[Neurons]
                : new Vector<float>[Neurons];
            neuronVecAccs.Fill(Vector<float>.Zero);

            for (; idx < _inputs.Length - Vector<float>.Count; idx += Vector<float>.Count)
            {
                var leftVec = Vector.LoadUnsafe(ref _inputs.UnsafeIndex(idx));
                for (var neuronIdx = 0; neuronIdx < Neurons; neuronIdx++)
                {
                    ref var vecAcc = ref neuronVecAccs.UnsafeIndex(neuronIdx);
                    var rightVec = Vector.LoadUnsafe(ref Unsafe.Add(ref weights, neuronIdx * Inputs + idx));

                    if (typeof(float) == typeof(double))
                    {
                        vecAcc = Vector.FusedMultiplyAdd(
                            leftVec.As<float, double>(),
                            rightVec.As<float, double>(),
                            vecAcc.As<float, double>()).As<double, float>();
                    }
                    else if (typeof(float) == typeof(float))
                    {
                        vecAcc = Vector.FusedMultiplyAdd(
                            leftVec.As<float, float>(),
                            rightVec.As<float, float>(),
                            vecAcc.As<float, float>()).As<float, float>();
                    }
                    else
                    {
                        vecAcc += leftVec * rightVec;
                    }
                }
            }

            var slowStart = idx;
            for (var neuronIdx = 0; neuronIdx < Neurons; neuronIdx++)
            {
                var acc = Vector.Sum(neuronVecAccs.UnsafeIndex(neuronIdx));

                for (idx = slowStart; idx < Inputs; idx++)
                {
                    var leftNum = _inputs.UnsafeIndex(idx);
                    var rightNum = Unsafe.Add(ref weights, neuronIdx * Inputs + idx);
                    acc += leftNum * rightNum;
                }

                _output!.UnsafeIndex(neuronIdx) = acc;
            }

            MathEx.Binary<float, BUPipeline<float, AddOp<float>, ReLUOp<float>>>(_output, _biases, _output);
        }
        else // Software fallback
        {
            var biases = _biases;

            for (var neuronIdx = 0; neuronIdx < Neurons; neuronIdx++)
            {
                var acc = 0f;

                for (var idx = 0; idx < Inputs; idx++)
                {
                    var input = _inputs!.UnsafeIndex(idx);
                    var weight = Unsafe.Add(ref weights, neuronIdx * Inputs + idx);
                    acc += input * weight;
                }

                _output!.UnsafeIndex(neuronIdx) = MathEx.ReLU(acc + biases!.UnsafeIndex(neuronIdx));
            }
        }
    }

    [Benchmark]
    public void ParallelDotButLinearReLUAndBias()
    {
        var neuronIdx = 0;
        ref float weights = ref _weights!.Ref();
        ref float bias = ref _biases!.Ref();
        ref float output = ref _output!.Ref();
        ref float outputEnd = ref _output!.UnsafeIndex(Neurons);
        while (Unsafe.IsAddressLessThan(ref output, ref outputEnd))
        {
            var dot = MathEx.Aggregate<float, MulOp<float>, AddOp<float>>(
                MemoryMarshal.CreateReadOnlySpan(ref weights, Inputs),
                _inputs);
            output = MathEx.ReLU(dot + bias);

            neuronIdx++;
            weights = ref Unsafe.Add(ref weights, Inputs);
            bias = ref Unsafe.Add(ref bias, 1);
            output = ref Unsafe.Add(ref output, 1);
        }
    }

    [Benchmark]
    public void ParallelDotReLUAndBias()
    {
        var neuronIdx = 0;
        ref float weights = ref _weights!.Ref();
        ref float output = ref _output!.Ref();
        ref float outputEnd = ref _output!.UnsafeIndex(Neurons);
        while (Unsafe.IsAddressLessThan(ref output, ref outputEnd))
        {
            output = MathEx.Aggregate<float, MulOp<float>, AddOp<float>>(
                MemoryMarshal.CreateReadOnlySpan(ref weights, Inputs),
                _inputs);

            neuronIdx++;
            weights = ref Unsafe.Add(ref weights, Inputs);
            output = ref Unsafe.Add(ref output, 1);
        }

        MathEx.Binary<float, BUPipeline<float, AddOp<float>, ReLUOp<float>>>(_output, _biases, _output);
    }

    private delegate void InferenceDelegate(ReadOnlySpan<float> inputs, Span<float> output);
    private void Run(InferenceDelegate inference) => inference(_inputs, _output);
}
