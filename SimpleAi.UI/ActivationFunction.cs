using System.Diagnostics.CodeAnalysis;

namespace SimpleAi.UI;

internal enum ActivationFunction
{
    [SuppressMessage(category: "ReSharper", checkId: "InconsistentNaming", Justification = "Class name.")]
    ReLU,
    Sigmoid,
    TanH,
    SoftMax,
}
