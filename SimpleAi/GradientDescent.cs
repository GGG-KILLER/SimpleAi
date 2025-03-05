using System.Numerics;
using System.Numerics.Tensors;

namespace SimpleAi;

internal static class GradientDescent
{
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

    public static Tensor<T> CalculateOutputLayerTrailingDerivatives<T, TCost>(
        Layer<T>                 layer,
        in ReadOnlyTensorSpan<T> expectedOutputs,
        in ReadOnlyTensorSpan<T> unactivatedOutputs,
        in ReadOnlyTensorSpan<T> actualOutputs) where T : IFloatingPoint<T> where TCost : ICostFunction<T>
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

        var costDerivatives       = TCost.Derivative(expectedOutputs, actualOutputs);
        var activationDerivatives = layer.CalculateActivationDerivatives(unactivatedOutputs);
        return Tensor.Multiply<T>(activationDerivatives, costDerivatives);
    }

    public static void UpdateLayerGradients<T>(
        Layer<T>                 layer,
        in ReadOnlyTensorSpan<T> inputs,
        in ReadOnlyTensorSpan<T> trailingDerivatives,
        Tensor<T>                weightCostGradients,
        Tensor<T>                biasCostGradients) where T : IFloatingPoint<T>
    {
        ArgumentNullException.ThrowIfNull(layer);
        if (!inputs.Lengths.SequenceEqual([layer.Inputs]))
            throw new ArgumentException(message: "Inputs do not match Layer's amount of inputs.", nameof(inputs));
        if (!trailingDerivatives.Lengths.SequenceEqual([layer.Neurons]))
            throw new ArgumentException(
                message: "Trailing derivatives does match Layer's amount of nodes.",
                nameof(trailingDerivatives));
        if (!weightCostGradients.Lengths.SequenceEqual(layer.Weights.Lengths))
            throw new ArgumentException(
                message: "Weight cost gradients' dimensions does not match layer's weights' dimensions.",
                nameof(weightCostGradients));
        if (!biasCostGradients.Lengths.SequenceEqual([layer.Neurons]))
            throw new ArgumentException(
                message: "Bias cost gradients' size does not match layer's neurons.",
                nameof(biasCostGradients));

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
                T weightDerivative = inputs[inputIdx] * nodeTrailingDerivatives;

                // Adds to the gradient costs as we're doing this for one of the data points, and we want the final result to be the average of all of them.
                weightCostGradients[nodeIdx, inputIdx] += weightDerivative;
            }
        }

        /*
         * The result of  ∂c    ∂z_n
         *               ---- = ---- x trailingDerivatives_n
         *               ∂b_n   ∂b_n
         * And, since z_n = a_n * w_n + b_n, nothing directly affects the bias, so it's a constant, so its derivative is one since the derivative of a constant is one.
         */
        Tensor.Add(biasCostGradients, trailingDerivatives, biasCostGradients);
    }
}
