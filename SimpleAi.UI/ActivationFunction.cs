using System.Diagnostics.CodeAnalysis;

namespace SimpleAi.UI;

internal enum ActivationFunction
{
    [SuppressMessage("ReSharper", "InconsistentNaming", Justification = "Class name.")]
    ReLU,
    Sigmoid,
    TanH,
    SoftMax,
}
