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
  Job-OJKLEP : .NET 9.0.1 (9.0.124.61010), X64 RyuJIT AVX2

OutlierMode=DontRemove  MemoryRandomization=True  RunStrategy=Throughput

```
| Method      | Inputs | Neurons | Mean              | Error            | StdDev           | Median            |
|------------ |------- |-------- |------------------:|-----------------:|-----------------:|------------------:|
| **FloatInfer**  | **5**      | **5**       |          **26.72 ns** |         **0.435 ns** |         **0.406 ns** |          **26.60 ns** |
| DoubleInfer | 5      | 5       |          28.50 ns |         0.595 ns |         0.708 ns |          28.41 ns |
| **FloatInfer**  | **5**      | **10**      |          **46.61 ns** |         **0.963 ns** |         **1.583 ns** |          **46.39 ns** |
| DoubleInfer | 5      | 10      |          48.32 ns |         0.962 ns |         0.900 ns |          48.05 ns |
| **FloatInfer**  | **5**      | **250**     |         **997.23 ns** |         **6.880 ns** |         **6.436 ns** |         **997.21 ns** |
| DoubleInfer | 5      | 250     |       1,028.78 ns |        20.097 ns |        35.198 ns |       1,021.12 ns |
| **FloatInfer**  | **5**      | **5000**    |      **19,008.55 ns** |       **167.770 ns** |       **156.932 ns** |      **18,997.45 ns** |
| DoubleInfer | 5      | 5000    |      19,025.51 ns |       262.949 ns |       245.963 ns |      19,111.78 ns |
| **FloatInfer**  | **5**      | **10000**   |      **38,582.86 ns** |       **747.398 ns** |       **971.829 ns** |      **38,442.78 ns** |
| DoubleInfer | 5      | 10000   |      38,524.98 ns |       746.736 ns |       829.996 ns |      38,321.02 ns |
| **FloatInfer**  | **10**     | **5**       |          **44.99 ns** |         **0.884 ns** |         **0.826 ns** |          **44.78 ns** |
| DoubleInfer | 10     | 5       |          30.99 ns |         0.665 ns |         1.580 ns |          30.84 ns |
| **FloatInfer**  | **10**     | **10**      |          **86.61 ns** |         **1.780 ns** |         **3.908 ns** |          **85.85 ns** |
| DoubleInfer | 10     | 10      |          48.36 ns |         1.009 ns |         1.657 ns |          47.88 ns |
| **FloatInfer**  | **10**     | **250**     |       **1,818.67 ns** |        **27.642 ns** |        **25.856 ns** |       **1,817.55 ns** |
| DoubleInfer | 10     | 250     |       1,208.24 ns |        24.166 ns |        24.817 ns |       1,206.38 ns |
| **FloatInfer**  | **10**     | **5000**    |      **35,598.32 ns** |       **410.918 ns** |       **384.373 ns** |      **35,617.37 ns** |
| DoubleInfer | 10     | 5000    |      63,851.54 ns |     1,872.786 ns |     5,521.951 ns |      59,842.12 ns |
| **FloatInfer**  | **10**     | **10000**   |      **71,153.08 ns** |       **883.817 ns** |       **826.723 ns** |      **70,972.79 ns** |
| DoubleInfer | 10     | 10000   |      59,645.51 ns |     1,885.919 ns |     5,560.674 ns |      58,044.90 ns |
| **FloatInfer**  | **250**    | **5**       |         **165.49 ns** |         **3.360 ns** |         **7.720 ns** |         **161.46 ns** |
| DoubleInfer | 250    | 5       |         309.59 ns |         6.251 ns |        15.097 ns |         314.63 ns |
| **FloatInfer**  | **250**    | **10**      |         **288.07 ns** |         **5.822 ns** |        **17.167 ns** |         **283.80 ns** |
| DoubleInfer | 250    | 10      |         638.24 ns |        60.798 ns |       179.263 ns |         539.76 ns |
| **FloatInfer**  | **250**    | **250**     |       **8,956.55 ns** |       **176.697 ns** |       **203.484 ns** |       **8,924.68 ns** |
| DoubleInfer | 250    | 250     |      20,237.00 ns |       498.458 ns |     1,469.714 ns |      20,950.85 ns |
| **FloatInfer**  | **250**    | **5000**    |     **301,232.30 ns** |     **6,002.626 ns** |    **13,301.404 ns** |     **298,923.04 ns** |
| DoubleInfer | 250    | 5000    |     706,984.51 ns |    14,062.680 ns |    35,020.957 ns |     711,942.04 ns |
| **FloatInfer**  | **250**    | **10000**   |     **685,006.92 ns** |    **13,431.967 ns** |    **20,104.338 ns** |     **689,578.32 ns** |
| DoubleInfer | 250    | 10000   |   1,919,711.78 ns |    56,890.746 ns |   167,743.661 ns |   1,959,482.26 ns |
| **FloatInfer**  | **5000**   | **5**       |       **2,921.08 ns** |        **62.540 ns** |       **184.400 ns** |       **2,858.06 ns** |
| DoubleInfer | 5000   | 5       |       6,168.83 ns |       642.958 ns |     1,895.776 ns |       5,746.03 ns |
| **FloatInfer**  | **5000**   | **10**      |       **5,324.65 ns** |       **110.905 ns** |       **327.007 ns** |       **5,273.80 ns** |
| DoubleInfer | 5000   | 10      |      10,988.81 ns |       227.231 ns |       669.996 ns |      11,150.05 ns |
| **FloatInfer**  | **5000**   | **250**     |     **207,070.99 ns** |     **8,357.076 ns** |    **24,641.029 ns** |     **219,256.36 ns** |
| DoubleInfer | 5000   | 250     |     448,823.79 ns |    13,298.455 ns |    39,210.799 ns |     463,805.41 ns |
| **FloatInfer**  | **5000**   | **5000**    |  **24,959,359.91 ns** |   **237,240.150 ns** |   **221,914.583 ns** |  **24,951,968.94 ns** |
| DoubleInfer | 5000   | 5000    |  39,368,677.02 ns |   592,820.975 ns |   554,525.108 ns |  39,269,695.15 ns |
| **FloatInfer**  | **5000**   | **10000**   |  **48,994,629.97 ns** |   **670,481.612 ns** |   **627,168.917 ns** |  **48,819,363.18 ns** |
| DoubleInfer | 5000   | 10000   |  76,597,654.33 ns | 1,507,099.550 ns | 2,390,419.787 ns |  76,619,919.50 ns |
| **FloatInfer**  | **10000**  | **5**       |       **5,884.99 ns** |       **116.742 ns** |       **224.923 ns** |       **5,790.41 ns** |
| DoubleInfer | 10000  | 5       |      11,951.79 ns |       238.823 ns |       460.130 ns |      11,755.84 ns |
| **FloatInfer**  | **10000**  | **10**      |      **11,386.46 ns** |       **208.012 ns** |       **194.575 ns** |      **11,435.93 ns** |
| DoubleInfer | 10000  | 10      |      23,093.21 ns |       372.315 ns |       348.264 ns |      23,102.00 ns |
| **FloatInfer**  | **10000**  | **250**     |     **447,721.28 ns** |    **10,333.381 ns** |    **30,468.209 ns** |     **457,481.25 ns** |
| DoubleInfer | 10000  | 250     |     969,094.84 ns |    31,406.461 ns |    92,602.665 ns |     974,054.58 ns |
| **FloatInfer**  | **10000**  | **5000**    |  **39,542,151.14 ns** |   **784,450.203 ns** | **1,511,369.340 ns** |  **39,236,183.81 ns** |
| DoubleInfer | 10000  | 5000    |  89,550,482.29 ns | 1,787,033.771 ns | 4,580,851.071 ns |  88,913,975.50 ns |
| **FloatInfer**  | **10000**  | **10000**   |  **79,549,483.59 ns** | **1,581,243.727 ns** | **2,769,421.054 ns** |  **79,257,133.57 ns** |
| DoubleInfer | 10000  | 10000   | 175,318,524.94 ns | 3,483,899.067 ns | 5,214,536.511 ns | 174,956,504.00 ns |
