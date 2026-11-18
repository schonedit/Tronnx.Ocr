using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Tronnx.Ocr.Detection;
using Tronnx.Ocr.Imaging;
using Tronnx.Ocr.Models;
using Tronnx.Ocr.Recognition;

namespace Tronnx.Ocr;

/// <summary>
/// Az OCR-motor: egyszer hozd létre, és használd újra (a modellek betöltése a drága rész).
/// Szálbiztos: több szálból is hívható párhuzamosan.
/// </summary>
/// <example>
/// <code>
/// using var engine = OcrEngine.Create(new LatinModelPack());
/// OcrPage page = engine.Recognize("scan.png");
/// Console.WriteLine(page.Text);
/// </code>
/// </example>
public sealed class OcrEngine : IDisposable
{
    private readonly DbTextDetector _detector;
    private readonly CtcTextRecognizer _recognizer;
    private readonly OcrEngineOptions _options;

    private OcrEngine(DbTextDetector detector, CtcTextRecognizer recognizer, OcrEngineOptions options, IOcrModelPack models)
    {
        _detector = detector;
        _recognizer = recognizer;
        _options = options;
        Models = models;
    }

    /// <summary>A betöltött modellcsomag.</summary>
    public IOcrModelPack Models { get; }

    public static OcrEngine Create(IOcrModelPack models, OcrEngineOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(models);
        options ??= new OcrEngineOptions();

        InferenceSession? detSession = null, recSession = null;
        try
        {
            using (var so = CreateSessionOptions(options))
            {
                detSession = new InferenceSession(models.LoadDetectionModel(), so);
                recSession = new InferenceSession(models.LoadRecognitionModel(), so);
            }
            var detector = new DbTextDetector(detSession, options);
            var recognizer = new CtcTextRecognizer(recSession, models.LoadDictionary(), options);
            return new OcrEngine(detector, recognizer, options, models);
        }
        catch
        {
            detSession?.Dispose();
            recSession?.Dispose();
            throw;
        }
    }

    private static SessionOptions CreateSessionOptions(OcrEngineOptions options)
    {
        var so = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
        };
        if (options.IntraOpThreads > 0)
            so.IntraOpNumThreads = options.IntraOpThreads;
        options.ConfigureSession?.Invoke(so);
        return so;
    }

    public OcrPage Recognize(string imagePath) => Recognize(ImageLoader.Load(imagePath));

    public OcrPage Recognize(Stream imageStream) => Recognize(ImageLoader.Load(imageStream));

    public OcrPage Recognize(BgrImage image)
    {
        ArgumentNullException.ThrowIfNull(image);
        var sw = Stopwatch.StartNew();

        var boxes = _detector.Detect(image);
        double detMs = sw.Elapsed.TotalMilliseconds;

        var lines = _recognizer.Recognize(image, boxes);
        double recMs = sw.Elapsed.TotalMilliseconds - detMs;

        var kept = lines
            .Where(l => l.Text.Length > 0 && l.Confidence >= _options.MinLineConfidence)
            .Select(l => l with { Box = l.Box.ClampTo(image.Width, image.Height) });
        var ordered = ReadingOrder.Sort(kept);

        return new OcrPage(image.Width, image.Height, ordered, new OcrTimings(detMs, recMs, sw.Elapsed.TotalMilliseconds));
    }

    /// <summary>Kényelmi metódus: csak a szöveg.</summary>
    public string GetText(string imagePath) => Recognize(imagePath).Text;

    public void Dispose()
    {
        _detector.Dispose();
        _recognizer.Dispose();
    }
}
