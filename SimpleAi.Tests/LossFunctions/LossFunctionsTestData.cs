namespace SimpleAi.Tests.LossFunctions;

internal static class LossFunctionsTestData
{
    public static TheoryData<int> InputSizes => [1, 2, 4, 8, 10, 16, 32, 64, 100, 128, 1000];

    public static int IterationCount => 1000;

    // This is as far as the float tests can go, precision-wise.
    public static float MeanSquaredErrorTolerance => 0.0005f;

    // This is all I can be bothered to get it down to, honestly.
    public static float CrossEntropyTolerance => 0.1f;
}
