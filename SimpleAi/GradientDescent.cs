using System.Numerics;
using SimpleAi.Math;

namespace SimpleAi;

internal static class GradientDescent
{
    public static int GetTrailingDerivativesBufferSize<T>(NeuralNetwork<T> neuralNetwork)
        where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(neuralNetwork);

        var layers = neuralNetwork.Layers;
        int size   = -1;

        for (var idx = 0; idx < layers.Length; idx++)
        {
            size = int.Max(size, layers.UnsafeIndex(idx).Outputs);
        }

        return size;
    }

    public static void CalculateHiddenLayerTrailingDerivatives<T>(
        Layer<T>        currentLayer,
        Layer<T>        nextLayer,
        ReadOnlySpan<T> inputs,
        ReadOnlySpan<T> activationInputs,
        Span<T>         activationDerivatives,
        Span<T>         trailingDerivatives) where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(currentLayer);
        if (inputs.Length != currentLayer.Inputs)
            throw new ArgumentException("Amount of inputs does not match Layer's.", nameof(inputs));
        if (activationInputs.Length != currentLayer.Outputs)
            throw new ArgumentException(
                "Amount of activation inputs does not match Layer's outputs.",
                nameof(activationInputs));

        if (trailingDerivatives.Length < currentLayer.Outputs)
            throw new ArgumentException(
                "Trailing derivatives cannot store the partial derivatives for this hidden layer.",
                nameof(trailingDerivatives));
        if (trailingDerivatives.Length < nextLayer.Outputs)
            throw new ArgumentException(
                "Trailing derivatives does not contain all partial derivatives for the nodes in the next layer.",
                nameof(trailingDerivatives));

        currentLayer.CalculateActivationDerivatives(activationInputs, activationDerivatives);

        // Calculates ∂z_{n+1}/∂a_n * trailingDerivatives_{n+1} 
        //              k
        // ∂z_{n+1}   \----  ∂z_{n+1,i}
        // -------- =  >     ----------
        //   ∂a_n     /----   ∂a_{n,i} 
        //             i=0
        for (var currentLayerNodeIdx = 0; currentLayerNodeIdx < currentLayer.Outputs; currentLayerNodeIdx++)
        {
            var currentNodeTrailingDerivative = T.Zero;
            // Calculates ∂z_{n+1,i}/∂a_{n,i} * trailingDerivatives{n+1,i}
            // Note: ∂z_{n+1,i}/∂a_{n,i} = w_{n+1,i}
            for (var nextLayerNodeIdx = 0; nextLayerNodeIdx < nextLayer.Outputs; nextLayerNodeIdx++)
                currentNodeTrailingDerivative +=
                    nextLayer.Weights[nodeIndex: nextLayerNodeIdx, inputIndex: currentLayerNodeIdx]
                    * trailingDerivatives[nextLayerNodeIdx];
            trailingDerivatives[currentLayerNodeIdx] = currentNodeTrailingDerivative;
        }

        //                          ∂a_n  ∂a_n   ∂z_{n+1}
        // trailingDerivatives_n = ---- = ---- x -------- x trailingDerivatives_{n+1}
        //                          ∂w_n  ∂z_n     ∂a_n
        MathEx.Binary<T, MulOp<T>>(
            trailingDerivatives[..currentLayer.Outputs],
            activationDerivatives[..currentLayer.Outputs],
            trailingDerivatives[..currentLayer.Outputs]);
    }

    public static void CalculateOutputLayerTrailingDerivatives<T, TCost>(
        Layer<T>        layer,
        ReadOnlySpan<T> expectedOutputs,
        ReadOnlySpan<T> activationInputs,
        ReadOnlySpan<T> actualOutputs,
        Span<T>         activationDerivatives,
        Span<T>         costDerivatives,
        Span<T>         trailingDerivatives) where T : unmanaged, INumber<T> where TCost : ICostFunction<T>
    {
        ArgumentNullException.ThrowIfNull(layer);
        // Input data needs to be the exact size.
        if (expectedOutputs.Length != layer.Outputs)
            throw new ArgumentException("Expected outputs has invalid size.", nameof(expectedOutputs));
        if (activationInputs.Length != layer.Outputs)
            throw new ArgumentException("Activation inputs has invalid size.", nameof(activationInputs));
        if (actualOutputs.Length != layer.Outputs)
            throw new ArgumentException("Actual outputs has invalid size.", nameof(actualOutputs));

        // Storage buffers can be larger, so we check if they can store at least the number of elements we need.
        if (activationDerivatives.Length < layer.Outputs)
            throw new ArgumentException("Activation derivatives too small.", nameof(activationDerivatives));
        if (costDerivatives.Length < layer.Outputs)
            throw new ArgumentException("Cost derivatives too small.", nameof(costDerivatives));
        if (trailingDerivatives.Length < layer.Outputs)
            throw new ArgumentException("Trailing derivatives too small.", nameof(trailingDerivatives));

        TCost.Derivative(expectedOutputs, actualOutputs, costDerivatives);
        layer.CalculateActivationDerivatives(activationInputs, activationDerivatives);
        MathEx.Binary<T, MulOp<T>>(activationDerivatives, costDerivatives, trailingDerivatives);
    }

    public static void UpdateLayerGradients<T>(
        Layer<T>        layer,
        ReadOnlySpan<T> inputs,
        ReadOnlySpan<T> trailingDerivatives,
        Weights<T>      weightCostGradients,
        Span<T>         biasCostGradients) where T : unmanaged, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(layer);
        if (inputs.Length != layer.Inputs)
            throw new ArgumentException("Inputs do not match Layer's amount of inputs.", nameof(inputs));

        // Caller can re-use the trailing derivatives storage, so it might be larger than we need, so check if it has at least the number of elements we need.
        if (trailingDerivatives.Length < layer.Outputs)
            throw new ArgumentException(
                "Trailing derivatives does match Layer's amount of nodes.",
                nameof(trailingDerivatives));

        // Output buffers can be larger, so we check if they can store at least the number of elements we need. 
        if (weightCostGradients.Length < layer.Inputs * layer.Outputs)
            throw new ArgumentException("Weight cost gradients is too small.", nameof(weightCostGradients));
        if (biasCostGradients.Length < layer.Outputs)
            throw new ArgumentException("Bias cost gradients is too small.", nameof(biasCostGradients));

        for (var nodeIdx = 0; nodeIdx < layer.Outputs; nodeIdx++)
        {
            T nodeTrailingDerivatives = trailingDerivatives.UnsafeIndex(nodeIdx);

            for (var inputIdx = 0; inputIdx < layer.Inputs; inputIdx++)
            {
                /*
                 * The result of    ∂c        ∂z_n     ∂a_n
                 *               -------- = -------- x ---- x trailingDerivatives_n
                 *               ∂w_{n,i}   ∂w_{n,i}   ∂z_n
                 */
                T weightDerivative = inputs.UnsafeIndex(inputIdx) * nodeTrailingDerivatives;

                // Adds to the gradient costs as we're doing this for one of the data points, and we want the final result to be the average of all of them.
                weightCostGradients[nodeIdx, inputIdx] += weightDerivative;
            }

            /*
             * The result of  ∂c    ∂z_n
             *               ---- = ---- x trailingDerivatives_n
             *               ∂b_n   ∂b_n
             * And, since z_n = a_n * w_n + b_n, nothing directly affects the bias, so it's a constant, so its derivative is one since the derivative of a constant is one.
             */
            T biasDerivative = T.One * nodeTrailingDerivatives;
            biasCostGradients.UnsafeIndex(nodeIdx) += biasDerivative;
        }
    }
}
