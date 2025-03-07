using System.Numerics;
using System.Numerics.Tensors;
using JetBrains.Annotations;

namespace SimpleAi;

[PublicAPI]
public static class OptimizationHelper
{
    [PublicAPI]
    public static Tensor<T> CalculateHiddenLayerTrailingDerivatives<T>(
        Layer<T>                 currentLayer,
        Layer<T>                 nextLayer,
        in ReadOnlyTensorSpan<T> inputs,
        in ReadOnlyTensorSpan<T> unactivatedInputs,
        in ReadOnlyTensorSpan<T> trailingDerivatives) where T : IFloatingPoint<T>
    {
        ArgumentNullException.ThrowIfNull(currentLayer);
        ArgumentNullException.ThrowIfNull(nextLayer);

        if (!inputs.Lengths.SequenceEqual([currentLayer.Inputs]))
            throw new ArgumentException(message: "Amount of inputs does not match Layer's.", nameof(inputs));
        if (!unactivatedInputs.Lengths.SequenceEqual([currentLayer.Neurons]))
            throw new ArgumentException(
                message: "Amount of activation function inputs does not match Layer's outputs.",
                nameof(unactivatedInputs));

        // Calculates ∂z_{n+1}/∂a_n * trailingDerivatives_{n+1} 
        //              k
        // ∂z_{n+1}   \----  ∂z_{n+1,i}
        // -------- =  >     ----------
        //   ∂a_n     /----   ∂a_{n,i} 
        //             i=0
        var currentTrailingDerivatives = Tensor.Create<T>([currentLayer.Neurons]);
        for (var currentLayerNeuron = 0; currentLayerNeuron < currentLayer.Neurons; currentLayerNeuron++)
        {
            T currentNodeTrailingDerivative = T.Zero;
            // Calculates ∂z_{n+1,i}/∂a_{n,i} * trailingDerivatives{n+1,i}
            // Note: ∂z_{n+1,i}/∂a_{n,i} = w_{n+1,i}
            for (var nextLayerNeuron = 0; nextLayerNeuron < nextLayer.Neurons; nextLayerNeuron++)
            {
                currentNodeTrailingDerivative += nextLayer.Weights[nextLayerNeuron, currentLayerNeuron]
                                                 * trailingDerivatives[nextLayerNeuron];
            }
            currentTrailingDerivatives[currentLayerNeuron] = currentNodeTrailingDerivative;
        }

        //                         ∂a_n   ∂a_n   ∂z_{n+1}
        // trailingDerivatives_n = ---- = ---- x -------- x trailingDerivatives_{n+1}
        //                         ∂w_n   ∂z_n     ∂a_n
        Tensor.Multiply<T>(
            currentTrailingDerivatives,
            currentLayer.CalculateActivationDerivatives(unactivatedInputs),
            currentTrailingDerivatives);

        return currentTrailingDerivatives;
    }

    [PublicAPI]
    public static Tensor<T> CalculateOutputLayerTrailingDerivatives<T, TCost>(
        Layer<T>                 layer,
        in ReadOnlyTensorSpan<T> expectedOutputs,
        in ReadOnlyTensorSpan<T> unactivatedOutputs,
        in ReadOnlyTensorSpan<T> actualOutputs) where T : IFloatingPoint<T> where TCost : ILossFunction<T>
    {
        ArgumentNullException.ThrowIfNull(layer);
        // Input data needs to be the exact size.
        if (!expectedOutputs.Lengths.SequenceEqual([layer.Neurons]))
            throw new ArgumentException(message: "Expected outputs has invalid size.", nameof(expectedOutputs));
        if (!unactivatedOutputs.Lengths.SequenceEqual([layer.Neurons]))
            throw new ArgumentException(
                message: "Activation function inputs has invalid size.",
                nameof(unactivatedOutputs));
        if (!actualOutputs.Lengths.SequenceEqual([layer.Neurons]))
            throw new ArgumentException(message: "Actual outputs has invalid size.", nameof(actualOutputs));

        // Simplification of the gradient for cross-entropy cases.
        if (layer.ActivationType.IsGenericType
            && typeof(TCost).IsGenericType
            && ((layer.ActivationType.GetGenericTypeDefinition() == typeof(Softmax<>)
                 && typeof(TCost).GetGenericTypeDefinition() == typeof(MultiClassCrossEntropy<>))
                || (layer.ActivationType.GetGenericTypeDefinition() == typeof(Sigmoid<>)
                    && typeof(TCost).GetGenericTypeDefinition() == typeof(BinaryCrossEntropy<>))))
        {
            return Tensor.Subtract(actualOutputs, expectedOutputs);
        }

        var costDerivatives       = TCost.Derivative(expectedOutputs, actualOutputs);
        var activationDerivatives = layer.CalculateActivationDerivatives(unactivatedOutputs);
        return Tensor.Multiply<T>(activationDerivatives, costDerivatives);
    }

    [PublicAPI]
    public static void CalculateLayerGradients<T>(
        Layer<T>                  layer,
        in  ReadOnlyTensorSpan<T> inputs,
        in  ReadOnlyTensorSpan<T> trailingDerivatives,
        out Tensor<T>             weightCostGradients,
        out Tensor<T>             biasCostGradients) where T : IFloatingPoint<T>
    {
        ArgumentNullException.ThrowIfNull(layer);
        if (!inputs.Lengths.SequenceEqual([layer.Inputs]))
            throw new ArgumentException(message: "Inputs do not match Layer's amount of inputs.", nameof(inputs));
        if (!trailingDerivatives.Lengths.SequenceEqual([layer.Neurons]))
            throw new ArgumentException(
                message: "Trailing derivatives does match Layer's amount of nodes.",
                nameof(trailingDerivatives));

        weightCostGradients = Tensor.Create<T>(layer.Weights.Lengths);
        biasCostGradients   = Tensor.Create<T>(layer.Biases.Lengths);

        for (nint nodeIdx = 0; nodeIdx < layer.Neurons; nodeIdx++)
        {
            T nodeTrailingDerivatives = trailingDerivatives[nodeIdx];

            for (nint inputIdx = 0; inputIdx < layer.Inputs; inputIdx++)
            {
                /*
                 * The result of    ∂c       ∂z_n      ∂a_n
                 *               -------- = -------- x ---- x trailingDerivatives_n
                 *               ∂w_{n,i}   ∂w_{n,i}   ∂z_n
                 */
                weightCostGradients[nodeIdx, inputIdx] = inputs[inputIdx] * nodeTrailingDerivatives;
            }
        }

        /*
         * The result of  ∂c    ∂z_n
         *               ---- = ---- x trailingDerivatives_n
         *               ∂b_n   ∂b_n
         * And, since z_n = a_n * w_n + b_n, nothing directly affects the bias, so it's a constant, so its derivative is one since the derivative of a constant is one.
         */
        trailingDerivatives.CopyTo(biasCostGradients);
    }
}
