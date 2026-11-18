using System.Reflection;

namespace Tronnx.Ocr.Models;

/// <summary>
/// Modellcsomag: detektor + felismerő + karakterszótár. Ez a bővítési pont a nyelvekhez:
/// a gyári latin modell (Tronnx.Ocr.Models.Latin) és a későbbi fizetős, finomhangolt magyar
/// modell (Tronnx.Ocr.Pro) is ezt valósítja meg – a motornak mindegy, honnan jön a modell.
/// </summary>
public interface IOcrModelPack
{
    /// <summary>Rövid azonosító, pl. "latin-ppocrv5-mobile".</summary>
    string Name { get; }

    /// <summary>Támogatott nyelvek ISO 639-1 kóddal (tájékoztató jellegű).</summary>
    IReadOnlyList<string> Languages { get; }

    byte[] LoadDetectionModel();

    byte[] LoadRecognitionModel();

    /// <summary>A felismerő osztályai a blank nélkül, sorrendben (a szótárfájl sorai).</summary>
    IReadOnlyList<string> LoadDictionary();
}

/// <summary>Fájlokból betöltött modellcsomag – saját vagy kísérleti modellekhez.</summary>
public sealed class FileModelPack : IOcrModelPack
{
    private readonly string _detectionPath, _recognitionPath, _dictionaryPath;

    public FileModelPack(string detectionModelPath, string recognitionModelPath, string dictionaryPath,
        string name = "custom", IReadOnlyList<string>? languages = null)
    {
        _detectionPath = detectionModelPath;
        _recognitionPath = recognitionModelPath;
        _dictionaryPath = dictionaryPath;
        Name = name;
        Languages = languages ?? Array.Empty<string>();
    }

    public string Name { get; }
    public IReadOnlyList<string> Languages { get; }
    public byte[] LoadDetectionModel() => File.ReadAllBytes(_detectionPath);
    public byte[] LoadRecognitionModel() => File.ReadAllBytes(_recognitionPath);
    public IReadOnlyList<string> LoadDictionary() => DictionaryParser.Parse(File.ReadAllText(_dictionaryPath));
}

/// <summary>
/// Szerelvénybe ágyazott modellcsomag alaposztálya. Az 1.x a modelleket minden indításkor
/// ideiglenes fájlba írta ki; itt közvetlenül a memóriából töltjük be őket.
/// </summary>
public abstract class EmbeddedModelPack : IOcrModelPack
{
    private readonly Assembly _assembly;
    private readonly string _detectionResource, _recognitionResource, _dictionaryResource;

    protected EmbeddedModelPack(Assembly assembly, string detectionResource, string recognitionResource, string dictionaryResource)
    {
        _assembly = assembly;
        _detectionResource = detectionResource;
        _recognitionResource = recognitionResource;
        _dictionaryResource = dictionaryResource;
    }

    public abstract string Name { get; }
    public abstract IReadOnlyList<string> Languages { get; }

    public virtual byte[] LoadDetectionModel() => ReadBytes(_detectionResource);
    public virtual byte[] LoadRecognitionModel() => ReadBytes(_recognitionResource);
    public virtual IReadOnlyList<string> LoadDictionary()
        => DictionaryParser.Parse(System.Text.Encoding.UTF8.GetString(ReadBytes(_dictionaryResource)));

    protected byte[] ReadBytes(string resource)
    {
        using var stream = _assembly.GetManifestResourceStream(resource)
            ?? throw new InvalidOperationException(
                $"Embedded resource '{resource}' not found in {_assembly.GetName().Name}. " +
                "Did you run tools/download-models before building?");
        var bytes = new byte[stream.Length];
        stream.ReadExactly(bytes);
        return bytes;
    }
}

internal static class DictionaryParser
{
    /// <summary>Soronként egy osztály. Az üres sorokat kihagyjuk, a szóközt tartalmazó sort nem.</summary>
    public static IReadOnlyList<string> Parse(string content)
    {
        var result = new List<string>();
        foreach (var raw in content.Split('\n'))
        {
            var line = raw.TrimEnd('\r');
            if (line.Length > 0)
                result.Add(line);
        }
        return result;
    }
}
