using Microsoft.ML.OnnxRuntime;

namespace Tronnx.Ocr;

/// <summary>Az OCR-motor beállításai. Az alapértékek dokumentumszkenekre (150–300 dpi) vannak hangolva.</summary>
public sealed class OcrEngineOptions
{
    /// <summary>
    /// A detektor bemenetének hosszabb oldala legfeljebb ennyi pixel. Nagyobb érték: apróbb betűket
    /// is megtalál, de lassabb. Mérés (300 dpi-s A4, 11 pt): 960-nál is hibátlan detektálás.
    /// </summary>
    public int DetectionMaxSideLength { get; set; } = 960;

    /// <summary>A valószínűségi térkép küszöbe (pixel = szöveg, ha p &gt; küszöb).</summary>
    public float DetectionThreshold { get; set; } = 0.3f;

    /// <summary>Egy doboz átlagos valószínűségének minimuma; ez alatt eldobjuk.</summary>
    public float BoxThreshold { get; set; } = 0.6f;

    /// <summary>A doboz kitágításának mértéke (a DB a betűk „magját" jósolja).</summary>
    public float UnclipRatio { get; set; } = 1.5f;

    /// <summary>Hány szövegsort ismerjen fel egy modellhívás. CPU-n 4–8 a jó tartomány.</summary>
    public int RecognitionBatchSize { get; set; } = 8;

    /// <summary>Az ennél kisebb bizonyosságú sorokat kihagyjuk az eredményből.</summary>
    public float MinLineConfidence { get; set; } = 0.5f;

    /// <summary>ONNX Runtime szálak modellhívásonként (0 = az ORT dönt, általában a fizikai magok száma).</summary>
    public int IntraOpThreads { get; set; }

    /// <summary>
    /// Bővítési pont a munkamenet-beállításokhoz – pl. a későbbi GPU-csomag itt adja hozzá
    /// a DirectML / CUDA végrehajtót. A beépített beállítások után fut.
    /// </summary>
    public Action<SessionOptions>? ConfigureSession { get; set; }
}
