using System.Diagnostics;
using Tronnx.Ocr;
using Tronnx.Ocr.Imaging;
using Tronnx.Ocr.Models;

// Használat:
//   tronnx-ocr <kép>                  → szöveg kiírása
//   tronnx-ocr <kép> --boxes          → soronként bizonyosság + doboz
//   tronnx-ocr <kép> --bench 10       → 10 futás, átlagos idők (bemelegítés után)
//   tronnx-ocr <kép> --threads 4      → ONNX Runtime szálak száma

if (args.Length == 0)
{
    Console.WriteLine("usage: tronnx-ocr <image> [--boxes] [--bench N] [--threads N] [--det-size N]");
    return 1;
}

string path = args[0];
bool showBoxes = args.Contains("--boxes");
int bench = IntArg("--bench", 0);
var options = new OcrEngineOptions
{
    IntraOpThreads = IntArg("--threads", 0),
    DetectionMaxSideLength = IntArg("--det-size", 960),
};

var sw = Stopwatch.StartNew();
using var engine = OcrEngine.Create(new LatinModelPack(), options);
Console.Error.WriteLine($"Models loaded in {sw.ElapsedMilliseconds} ms ({engine.Models.Name})");

sw.Restart();
BgrImage image = ImageLoader.Load(path);
Console.Error.WriteLine($"Image {image.Width}x{image.Height} decoded in {sw.ElapsedMilliseconds} ms");

var page = engine.Recognize(image); // az első futás bemelegítés is (ORT lazy init)

if (bench > 0)
{
    double det = 0, rec = 0, total = 0;
    for (int i = 0; i < bench; i++)
    {
        var p = engine.Recognize(image);
        det += p.Timings.DetectionMs;
        rec += p.Timings.RecognitionMs;
        total += p.Timings.TotalMs;
    }
    Console.WriteLine($"{bench} runs, avg: detection {det / bench:F0} ms, recognition {rec / bench:F0} ms, total {total / bench:F0} ms, lines {page.Lines.Count}");
    return 0;
}

if (showBoxes)
{
    foreach (var line in page.Lines)
    {
        var b = line.Box;
        Console.WriteLine($"{line.Confidence:F2}  [{b.Left:F0},{b.Top:F0} {b.Width:F0}x{b.Height:F0} {b.AngleDegrees:F1}°]  {line.Text}");
    }
}
else
{
    Console.WriteLine(page.Text);
}
Console.Error.WriteLine($"detection {page.Timings.DetectionMs:F0} ms, recognition {page.Timings.RecognitionMs:F0} ms, {page.Lines.Count} lines");
return 0;

int IntArg(string name, int fallback)
{
    int i = Array.IndexOf(args, name);
    return i >= 0 && i + 1 < args.Length && int.TryParse(args[i + 1], out int v) ? v : fallback;
}
