# SimpleAi

A library to train and run Neural Networks on CPU using vectorization and as many optimizations as I can be bothered to implement.

This is most definitely **NOT PRODUCTION READY NOR WILL IT EVER BE**. If you want to actually create and run neural networks, use something like ML.NET, PyTorch or any other reputable AI library.

Was made while watching [Sebastian Lague](https://www.youtube.com/@SebastianLague)'s "[How to Create a Neural Network (and Train it to Identify Doodles)](https://youtu.be/hfMk-kjRv4c)" video and implementing my own optimizations on top of it.

## Performance

The following are **single-layer** inference results:

```

BenchmarkDotNet v0.14.0, NixOS 25.05 (Warbler)
AMD Ryzen 9 5950X, 1 CPU, 32 logical and 16 physical cores
.NET SDK 9.0.102
  [Host]     : .NET 9.0.1 (9.0.124.61010), X64 RyuJIT AVX2
  Job-YNSNNL : .NET 9.0.1 (9.0.124.61010), X64 RyuJIT AVX2

OutlierMode=DontRemove  MemoryRandomization=True  RunStrategy=Throughput

```
| Method      | Inputs | Neurons | Mean             | Error          | StdDev         | Median           |
|------------ |------- |-------- |-----------------:|---------------:|---------------:|-----------------:|
| **FloatInfer**  | **5**      | **5**       |         **24.62 ns** |       **0.425 ns** |       **0.398 ns** |         **24.61 ns** |
| DoubleInfer | 5      | 5       |         21.15 ns |       0.472 ns |       1.057 ns |         20.86 ns |
| **FloatInfer**  | **5**      | **10**      |         **30.39 ns** |       **0.654 ns** |       **0.851 ns** |         **30.15 ns** |
| DoubleInfer | 5      | 10      |         29.71 ns |       0.630 ns |       0.820 ns |         29.64 ns |
| **FloatInfer**  | **5**      | **250**     |        **519.59 ns** |      **10.100 ns** |      **12.773 ns** |        **517.69 ns** |
| DoubleInfer | 5      | 250     |        468.53 ns |       9.366 ns |       8.761 ns |        471.09 ns |
| **FloatInfer**  | **5**      | **5000**    |     **14,114.77 ns** |     **229.577 ns** |     **214.746 ns** |     **14,106.97 ns** |
| DoubleInfer | 5      | 5000    |      9,439.28 ns |     182.249 ns |     230.487 ns |      9,366.06 ns |
| **FloatInfer**  | **5**      | **10000**   |     **19,369.39 ns** |     **368.659 ns** |     **344.844 ns** |     **19,402.86 ns** |
| DoubleInfer | 5      | 10000   |     17,079.96 ns |     325.573 ns |     399.833 ns |     17,026.03 ns |
| **FloatInfer**  | **10**     | **5**       |         **21.82 ns** |       **0.471 ns** |       **0.561 ns** |         **21.80 ns** |
| DoubleInfer | 10     | 5       |         21.13 ns |       0.434 ns |       0.406 ns |         21.12 ns |
| **FloatInfer**  | **10**     | **10**      |         **31.30 ns** |       **0.604 ns** |       **0.565 ns** |         **31.21 ns** |
| DoubleInfer | 10     | 10      |         34.27 ns |       0.736 ns |       1.859 ns |         33.87 ns |
| **FloatInfer**  | **10**     | **250**     |        **574.08 ns** |      **11.488 ns** |      **23.466 ns** |        **566.93 ns** |
| DoubleInfer | 10     | 250     |        646.83 ns |      12.916 ns |      32.876 ns |        636.04 ns |
| **FloatInfer**  | **10**     | **5000**    |     **11,091.25 ns** |     **216.813 ns** |     **362.245 ns** |     **10,987.26 ns** |
| DoubleInfer | 10     | 5000    |     12,041.24 ns |     239.737 ns |     478.781 ns |     11,909.16 ns |
| **FloatInfer**  | **10**     | **10000**   |     **22,585.47 ns** |     **406.740 ns** |     **380.465 ns** |     **22,506.14 ns** |
| DoubleInfer | 10     | 10000   |     23,632.27 ns |     449.169 ns |     480.606 ns |     23,617.60 ns |
| **FloatInfer**  | **250**    | **5**       |         **96.23 ns** |       **1.904 ns** |       **1.781 ns** |         **95.85 ns** |
| DoubleInfer | 250    | 5       |        171.71 ns |       3.488 ns |       3.425 ns |        171.13 ns |
| **FloatInfer**  | **250**    | **10**      |        **169.03 ns** |       **1.599 ns** |       **1.495 ns** |        **168.90 ns** |
| DoubleInfer | 250    | 10      |        320.44 ns |       4.817 ns |       4.506 ns |        319.51 ns |
| **FloatInfer**  | **250**    | **250**     |      **4,049.51 ns** |      **80.446 ns** |      **92.642 ns** |      **4,046.57 ns** |
| DoubleInfer | 250    | 250     |      8,342.09 ns |     162.653 ns |     211.495 ns |      8,267.20 ns |
| **FloatInfer**  | **250**    | **5000**    |     **83,599.04 ns** |   **1,652.748 ns** |   **1,903.308 ns** |     **83,018.64 ns** |
| DoubleInfer | 250    | 5000    |    167,425.15 ns |   3,297.983 ns |   5,510.190 ns |    167,106.22 ns |
| **FloatInfer**  | **250**    | **10000**   |    **165,137.41 ns** |   **1,707.159 ns** |   **1,596.877 ns** |    **165,438.65 ns** |
| DoubleInfer | 250    | 10000   |    410,249.95 ns |  17,028.496 ns |  50,208.909 ns |    401,453.34 ns |
| **FloatInfer**  | **5000**   | **5**       |      **2,035.08 ns** |      **29.089 ns** |      **27.210 ns** |      **2,032.33 ns** |
| DoubleInfer | 5000   | 5       |      4,093.68 ns |      63.361 ns |      59.268 ns |      4,101.69 ns |
| **FloatInfer**  | **5000**   | **10**      |      **4,087.30 ns** |      **60.297 ns** |      **56.402 ns** |      **4,097.69 ns** |
| DoubleInfer | 5000   | 10      |      8,325.69 ns |      75.880 ns |      70.978 ns |      8,321.81 ns |
| **FloatInfer**  | **5000**   | **250**     |    **103,207.72 ns** |   **1,061.155 ns** |     **992.605 ns** |    **103,158.99 ns** |
| DoubleInfer | 5000   | 250     |    208,241.36 ns |   3,030.094 ns |   2,834.352 ns |    208,580.49 ns |
| **FloatInfer**  | **5000**   | **5000**    |  **3,437,112.44 ns** |  **99,370.284 ns** | **292,995.545 ns** |  **3,433,532.52 ns** |
| DoubleInfer | 5000   | 5000    |  7,125,344.72 ns | 139,737.031 ns | 166,347.009 ns |  7,082,096.05 ns |
| **FloatInfer**  | **5000**   | **10000**   |  **6,984,306.12 ns** | **139,156.703 ns** | **356,711.856 ns** |  **7,000,160.52 ns** |
| DoubleInfer | 5000   | 10000   | 14,334,050.01 ns | 279,964.084 ns | 383,218.031 ns | 14,253,190.48 ns |
| **FloatInfer**  | **10000**  | **5**       |      **4,159.55 ns** |      **43.135 ns** |      **40.349 ns** |      **4,150.93 ns** |
| DoubleInfer | 10000  | 5       |      8,484.05 ns |     152.039 ns |     142.217 ns |      8,474.12 ns |
| **FloatInfer**  | **10000**  | **10**      |      **8,509.36 ns** |     **110.312 ns** |     **103.186 ns** |      **8,510.43 ns** |
| DoubleInfer | 10000  | 10      |     16,830.46 ns |     147.161 ns |     137.655 ns |     16,806.13 ns |
| **FloatInfer**  | **10000**  | **250**     |    **210,682.00 ns** |   **2,301.564 ns** |   **2,152.884 ns** |    **211,085.23 ns** |
| DoubleInfer | 10000  | 250     |    462,035.39 ns |   9,201.194 ns |  27,129.929 ns |    453,170.41 ns |
| **FloatInfer**  | **10000**  | **5000**    |  **6,919,191.04 ns** | **137,917.955 ns** | **241,552.192 ns** |  **6,928,496.91 ns** |
| DoubleInfer | 10000  | 5000    | 14,084,140.05 ns | 273,118.745 ns | 408,791.312 ns | 14,106,132.63 ns |
| **FloatInfer**  | **10000**  | **10000**   | **13,790,495.12 ns** | **269,662.411 ns** | **443,063.111 ns** | **13,820,413.44 ns** |
| DoubleInfer | 10000  | 10000   | 28,274,879.96 ns | 541,177.381 ns | 555,749.327 ns | 28,122,404.72 ns |
