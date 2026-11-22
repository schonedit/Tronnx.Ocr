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

    /// <summary>
    /// Hány szövegsort ismerjen fel egy modellhívás. Mérés: 6 magos gépen 1–8 között nincs érdemi
    /// különbség, 2 magos gépen a 8-as köteg ~25%-kal lassabb a 2–4-esnél, ezért az alap 4.
    /// </summary>
    public int RecognitionBatchSize { get; set; } = 4;

    /// <summary>Az ennél kisebb bizonyosságú sorokat kihagyjuk az eredményből.</summary>
    public float MinLineConfidence { get; set; } = 0.5f;

    /// <summary>
    /// Hány szövegsor-köteget ismerjen fel párhuzamosan. 1 (alap) = soros feldolgozás, az ONNX Runtime
    /// belső szálaival; 0 = a logikai processzorok száma.
    /// Mérés (6 mag / 12 logikai, 53 soros A4): soros 1550 ms, párhuzamos (6) 1890 ms – sok sornál az ORT
    /// belső szálkezelése jobb. Csak kevés, nagyon eltérő hosszúságú sornál érdemes kipróbálni.
    /// </summary>
    public int RecognitionParallelism { get; set; } = 1;

    /// <summary>
    /// ONNX Runtime szálak modellhívásonként (0 = automatikus). Automatikus módban a detektor az ORT
    /// alapértékét kapja, a felismerő párhuzamos módban 1 szálat (a párhuzamosság a sorok között van).
    /// </summary>
    public int IntraOpThreads { get; set; }

    /// <summary>
    /// Bővítési pont a munkamenet-beállításokhoz – pl. a későbbi GPU-csomag itt adja hozzá
    /// a DirectML / CUDA végrehajtót. A beépített beállítások után fut.
    /// </summary>
    public Action<SessionOptions>? ConfigureSession { get; set; }
}
