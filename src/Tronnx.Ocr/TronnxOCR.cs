using Tronnx.Ocr.Models;

namespace Tronnx.Ocr;

/// <summary>
/// Az 1.x-es API megtartva, hogy a meglévő felhasználók kódja frissítés után is forduljon.
/// Új kódhoz az <see cref="OcrEngine"/>-t ajánljuk (dobozok, szavak, bizonyosság, időmérés).
/// A PDF-es metódusok (ToSearchablePdf, GetHighlightedPdf) a következő szakaszban jönnek vissza.
/// </summary>
public class TronnxOCR : IDisposable
{
    private readonly OcrEngine _engine;

    public TronnxOCR() : this(null)
    {
    }

    public TronnxOCR(OcrEngineOptions? options)
    {
        _engine = OcrEngine.Create(new LatinModelPack(), options);
    }

    /// <summary>A mögöttes motor, ha a részletes eredmény is kell.</summary>
    public OcrEngine Engine => _engine;

    /// <summary>A képen felismert teljes szöveg.</summary>
    public string GetText(string imagePath) => _engine.GetText(imagePath);

    public void Dispose()
    {
        _engine.Dispose();
        GC.SuppressFinalize(this);
    }
}
