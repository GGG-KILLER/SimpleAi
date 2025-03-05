using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading.Tasks;

namespace SimpleAi.UI.IO;

internal static class ModelSerializer
{
    public static async Task SaveModelAsync<T>(NeuralNetwork<T> neuralNetwork, Stream stream)
        where T : IFloatingPoint<T>
    {
        await using var writer = new Utf8JsonWriter(stream);
        writer.WriteStartObject();
        {
            writer.WriteNumber(utf8PropertyName: "version"u8, value: 2.0);
            writer.WriteString(utf8PropertyName: "numberType"u8, typeof(T).FullName);
            writer.WriteStartArray(utf8PropertyName: "layers"u8);
            {
                for (var index = 0; index < neuralNetwork.Layers.Length; index++)
                {
                    await writer.FlushAsync();
                    Layer<T> layer = neuralNetwork.Layers[index];

                    writer.WriteStartObject();
                    writer.WriteString(utf8PropertyName: "activation"u8, layer.GetType().GetGenericArguments()[1].Name);
                    writer.WriteNumber(utf8PropertyName: "inputs"u8, layer.Inputs);
                    writer.WriteNumber(utf8PropertyName: "neurons"u8, layer.Neurons);
                    await writer.FlushAsync();
                    writer.WriteStartArray(utf8PropertyName: "biases"u8);
                    foreach (var bias in layer.Biases)
                    {
                        if (typeof(T) == typeof(float))
                            writer.WriteNumberValue(Unsafe.BitCast<T, float>(bias));
                        else if (typeof(T) == typeof(double)) writer.WriteNumberValue(Unsafe.BitCast<T, double>(bias));
                    }
                    await writer.FlushAsync();
                    writer.WriteEndArray();
                    writer.WriteStartArray(utf8PropertyName: "weights"u8);
                    foreach (var weight in layer.Weights)
                    {
                        if (typeof(T) == typeof(float))
                            writer.WriteNumberValue(Unsafe.BitCast<T, float>(weight));
                        else if (typeof(T) == typeof(double))
                            writer.WriteNumberValue(Unsafe.BitCast<T, double>(weight));
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
