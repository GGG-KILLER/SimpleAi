using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading.Tasks;

namespace SimpleAi.UI.IO;

internal static class ModelSerializer
{
    public static async Task SaveModelAsync<T>(NeuralNetwork<T> neuralNetwork, Stream stream)
        where T : unmanaged, INumber<T>
    {
        await using var writer = new Utf8JsonWriter(stream);

        writer.WriteStartObject();
        {
            writer.WriteNumber("version"u8, 1.0);
            writer.WriteString("numberType"u8, typeof(T).FullName);
            writer.WriteStartArray("layers"u8);
            {
                for (var index = 0; index < neuralNetwork.Layers.Length; index++)
                {
                    await writer.FlushAsync();
                    Layer<T> layer = neuralNetwork.Layers[index];

                    writer.WriteStartObject();
                    writer.WriteString("activation"u8, layer.GetType().GetGenericArguments()[1].Name);
                    writer.WriteNumber("inputs"u8, layer.Inputs);
                    writer.WriteNumber("outputs"u8, layer.Outputs);
                    await writer.FlushAsync();
                    writer.WriteStartArray("biases"u8);
                    for (var biasIndex = 0; biasIndex < layer.Biases.Length; biasIndex++)
                    {
                        if (biasIndex % 10 == 0) await writer.FlushAsync();
                        T bias = layer.Biases[biasIndex];
                        if (typeof(T) == typeof(decimal))
                            writer.WriteNumberValue(Unsafe.BitCast<T, decimal>(bias));
                        else if (typeof(T) == typeof(double))
                            writer.WriteNumberValue(Unsafe.BitCast<T, double>(bias));
                        else if (typeof(T) == typeof(float))
                            writer.WriteNumberValue(Unsafe.BitCast<T, float>(bias));
                        else if (typeof(T) == typeof(sbyte))
                            writer.WriteNumberValue(Unsafe.BitCast<T, sbyte>(bias));
                        else if (typeof(T) == typeof(uint))
                            writer.WriteNumberValue(Unsafe.BitCast<T, uint>(bias));
                        else if (typeof(T) == typeof(ulong)) writer.WriteNumberValue(Unsafe.BitCast<T, ulong>(bias));
                    }
                    await writer.FlushAsync();
                    writer.WriteEndArray();
                    writer.WriteStartArray("weights"u8);
                    for (var weightIndex = 0; weightIndex < layer.Weights.Length; weightIndex++)
                    {
                        if (weightIndex % 10 == 0) await writer.FlushAsync();
                        T weight = layer.Weights[weightIndex];
                        if (typeof(T) == typeof(decimal))
                            writer.WriteNumberValue(Unsafe.BitCast<T, decimal>(weight));
                        else if (typeof(T) == typeof(double))
                            writer.WriteNumberValue(Unsafe.BitCast<T, double>(weight));
                        else if (typeof(T) == typeof(float))
                            writer.WriteNumberValue(Unsafe.BitCast<T, float>(weight));
                        else if (typeof(T) == typeof(sbyte))
                            writer.WriteNumberValue(Unsafe.BitCast<T, sbyte>(weight));
                        else if (typeof(T) == typeof(uint))
                            writer.WriteNumberValue(Unsafe.BitCast<T, uint>(weight));
                        else if (typeof(T) == typeof(ulong)) writer.WriteNumberValue(Unsafe.BitCast<T, ulong>(weight));
                    }
                    await writer.FlushAsync();
                    writer.WriteEndArray();
                    writer.WriteEndObject();
                }
            }
            writer.WriteEndArray();
        }
        writer.WriteEndObject();
        await writer.FlushAsync();
    }
}
