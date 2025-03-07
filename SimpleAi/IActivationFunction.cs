using System.Numerics;
using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi;

/// <summary>The interface for activation functions.</summary>
/// <typeparam name="T">The numeric type accepted by the activation function.</typeparam>
[PublicAPI]
public interface IActivationFunction<T>
{
    /// <summary>Executes the activation function on a set of <paramref name="inputs" />, return its results.</summary>
    /// <param name="inputs">The inputs to the activation function.</param>
    /// <returns>The activated results.</returns>
    [PublicAPI]
    static abstract Tensor<T> Activate(in ReadOnlyTensorSpan<T> inputs);

    /// <summary>
    ///     Runs the provided input <paramref name="inputs" /> through the derivative of the activation function,
    ///     returning its results.
    /// </summary>
    /// <param name="inputs">The inputs to pass through the derivative of the activation function.</param>
    /// <returns>The derivative of the activation function.</returns>
    [PublicAPI]
    static abstract Tensor<T> Derivative(in ReadOnlyTensorSpan<T> inputs);
}

/// <summary>
///     <c>S(x) = 1 / (1 + exp(-x))</c>
/// </summary>
/// <typeparam name="T"><inheritdoc /></typeparam>
[PublicAPI]
public readonly struct Sigmoid<T> : IActivationFunction<T> where T : IExponentialFunctions<T>
{
    /// <inheritdoc />
    public static Tensor<T> Activate(in ReadOnlyTensorSpan<T> inputs) => Tensor.Sigmoid(inputs);

    /// <inheritdoc />
    public static Tensor<T> Derivative(in ReadOnlyTensorSpan<T> inputs)
    {
        Tensor<T> activated = Activate(inputs);
        return Tensor.Multiply<T>(activated, Tensor.Subtract(T.One, activated));
    }
}

/// <summary>
///     <c>tanh(x) = (e^{2x} - 1)/(e^{2x} + 1)</c>
/// </summary>
/// <typeparam name="T"><inheritdoc /></typeparam>
[PublicAPI]
public readonly struct TanH<T> : IActivationFunction<T> where T : IExponentialFunctions<T>
{
    /// <inheritdoc />
    public static Tensor<T> Activate(in ReadOnlyTensorSpan<T> inputs)
    {
        Tensor<T> e2X = Tensor.Exp<T>(Tensor.Multiply(inputs, T.One + T.One));
        return Tensor.Divide<T>(Tensor.Subtract(e2X, T.One), Tensor.Add(e2X, T.One));
    }

    /// <inheritdoc />
    public static Tensor<T> Derivative(in ReadOnlyTensorSpan<T> inputs)
    {
        Tensor<T> tanhX = Activate(inputs);
        return Tensor.Subtract(T.One, Tensor.Multiply<T>(tanhX, tanhX));
    }
}

/// <summary>
///     <c>ReLU(x) = max(x, 0)</c>
/// </summary>
/// <typeparam name="T"><inheritdoc /></typeparam>
[PublicAPI]
public readonly struct ReLu<T> : IActivationFunction<T> where T : INumber<T> // Tensor.Max, T.Zero, T.One, operator >
{
    /// <inheritdoc />
    public static Tensor<T> Activate(in ReadOnlyTensorSpan<T> inputs) => Tensor.Max(inputs, T.Zero);

    /// <inheritdoc />
    public static Tensor<T> Derivative(in ReadOnlyTensorSpan<T> inputs)
    {
        Tensor<T>                        outputs           = Tensor.Create<T>(inputs.Lengths);
        TensorSpan<T>.Enumerator         outputsEnumerator = outputs.AsTensorSpan().GetEnumerator();
        ReadOnlyTensorSpan<T>.Enumerator inputsEnumerator  = inputs.GetEnumerator();
        while (inputsEnumerator.MoveNext() && outputsEnumerator.MoveNext())
        {
            outputsEnumerator.Current = inputsEnumerator.Current > T.Zero ? T.One : T.Zero;
        }
        return outputs;
    }
}

/// <summary>
///     <c>SiLU(x) = x / (1 + exp(-x))</c>
/// </summary>
/// <typeparam name="T"><inheritdoc /></typeparam>
[PublicAPI]
public readonly struct SiLu<T> : IActivationFunction<T>
    where T : INumberBase<T>, // T.One
    IExponentialFunctions<T>  // Tensor.Exp 
{
    /// <inheritdoc />
    public static Tensor<T> Activate(in ReadOnlyTensorSpan<T> inputs)
    {
        Tensor<T> one = Tensor.Create<T>(inputs.Lengths);
        one.Fill(T.One);

        // inputs / (1 + exp(-inputs))
        return Tensor.Divide(inputs, Tensor.Add<T>(one, Tensor.Exp<T>(Tensor.Negate(inputs))));
    }

    /// <inheritdoc />
    public static Tensor<T> Derivative(in ReadOnlyTensorSpan<T> inputs)
    {
        Tensor<T> one = Tensor.Create<T>(inputs.Lengths);
        one.Fill(T.One);

        Tensor<T> sig = Activate(inputs);
        // inputs * sig * (1 - sig) + sig
        return Tensor.Add<T>(Tensor.Multiply<T>(Tensor.Multiply(inputs, sig), Tensor.Subtract<T>(one, sig)), sig);
    }
}

/// <summary>
///     <c>S(x) = exp(x)/sum(exp(x))</c>
/// </summary>
/// <remarks>
/// <b>HAS NOT BEEN IMPLEMENTED ENTIRELY:</b> Usage of this activation function for training in cases other than as the function of the output layer with the <see cref="MultiClassCrossEntropy{T}"/> loss function.
/// </remarks>
/// <typeparam name="T"><inheritdoc /></typeparam>
[PublicAPI]
public readonly struct Softmax<T> : IActivationFunction<T> where T : IExponentialFunctions<T>
{
    /// <inheritdoc />
    public static Tensor<T> Activate(in ReadOnlyTensorSpan<T> inputs) => Tensor.SoftMax(inputs);

    /// <inheritdoc />
    public static Tensor<T> Derivative(in ReadOnlyTensorSpan<T> inputs)
        => throw new NotSupportedException("Softmax usage not supported.");
}
