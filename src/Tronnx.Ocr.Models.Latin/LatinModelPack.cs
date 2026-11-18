namespace Tronnx.Ocr.Models;

/// <summary>
/// A gyári modellcsomag: PP-OCRv5 mobile detektor + latin írású felismerő (32 nyelv, köztük a magyar).
/// Ingyenes, Apache 2.0. A magyar ékezeteket (ő, ű) ismeri, de néha ö/ü-re téveszti –
/// a pontosabb, finomhangolt magyar modell a Tronnx.Ocr.Pro csomagban lesz.
/// </summary>
public sealed class LatinModelPack : EmbeddedModelPack
{
    private static readonly string[] SupportedLanguages =
    [
        "en", "de", "fr", "es", "it", "pt", "nl", "pl", "cs", "sk", "hu", "hr", "bs", "sr", "sl",
        "da", "no", "sv", "is", "et", "lt", "sq", "cy", "ga", "id", "ms", "af", "oc", "uz", "mi", "tr", "la",
    ];

    public LatinModelPack()
        : base(typeof(LatinModelPack).Assembly,
            "Tronnx.Models.Latin.det.onnx",
            "Tronnx.Models.Latin.rec.onnx",
            "Tronnx.Models.Latin.dict.txt")
    {
    }

    public override string Name => "latin-ppocrv5-mobile";

    public override IReadOnlyList<string> Languages => SupportedLanguages;
}
