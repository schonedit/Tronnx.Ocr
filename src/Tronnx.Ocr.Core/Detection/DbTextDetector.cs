using System.Buffers;
using Microsoft.ML.OnnxRuntime;
using Tronnx.Ocr.Imaging;

namespace Tronnx.Ocr.Detection;

/// <summary>
/// Szövegdetektálás a PP-OCRv5 DB modellel. Bemenet: [1, 3, H, W] (H, W 32 többszöröse),
/// kimenet: [1, 1, H, W] valószínűségi térkép (minden pixelre: mennyire „szöveg").
/// </summary>
internal sealed class DbTextDetector : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string _outputName;
    private readonly int _maxSideLength;
    private readonly DbPostProcessor _post;

    public DbTextDetector(InferenceSession session, OcrEngineOptions options)
    {
        _session = session;
        _inputName = session.InputMetadata.Keys.First();
        _outputName = session.OutputMetadata.Keys.First();
        _maxSideLength = options.DetectionMaxSideLength;
        _post = new DbPostProcessor(options.DetectionThreshold, options.BoxThreshold, options.UnclipRatio);
    }

    public List<Quad> Detect(BgrImage image)
    {
        // --- Előfeldolgozás: a hosszabb oldal legfeljebb _maxSideLength, mindkét oldal 32 többszöröse ---
        float ratio = MathF.Min(1f, (float)_maxSideLength / Math.Max(image.Width, image.Height));
        int dw = Math.Max(32, (int)MathF.Round(image.Width * ratio / 32f) * 32);
        int dh = Math.Max(32, (int)MathF.Round(image.Height * ratio / 32f) * 32);
        float sx = (float)dw / image.Width, sy = (float)dh / image.Height;

        int plane = dw * dh;
        float[] buffer = ArrayPool<float>.Shared.Rent(plane * 3);
        try
        {
            var tensor = buffer.AsSpan(0, plane * 3);
            Sampler.SampleToChw(image,
                origin: new OcrPoint(0, 0),
                du: new OcrPoint(1f / sx, 0),
                dv: new OcrPoint(0, 1f / sy),
                dw, dh, tensor, dw, plane, ChannelNorm.ImageNet);

            using var input = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance,
                buffer.AsMemory(0, plane * 3), new long[] { 1, 3, dh, dw });
            using var runOptions = new RunOptions();
            using var outputs = _session.Run(runOptions, new[] { _inputName }, new[] { input }, new[] { _outputName });

            var probMap = outputs[0].GetTensorDataAsSpan<float>();
            var shape = outputs[0].GetTensorTypeAndShape().Shape; // [1, 1, H, W]
            int ph = (int)shape[2], pw = (int)shape[3];

            var boxes = _post.Process(probMap, pw, ph);

            // --- Visszaskálázás a forráskép koordinátáiba ---
            // (A térkép mérete egyezik a bemenetével, de a biztonság kedvéért külön skálázunk.)
            float kx = (float)pw / dw * sx, ky = (float)ph / dh * sy;
            var quads = new List<Quad>(boxes.Count);
            foreach (var b in boxes)
                quads.Add(ToQuad(b, kx, ky));
            return quads;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    private static Quad ToQuad(RotatedBox b, float kx, float ky)
    {
        var c = new OcrPoint(b.CenterX / kx, b.CenterY / ky);
        // fél-vektorok: a = u · hw, bv = v · hh, ahol v = (-uy, ux) (képen lefelé mutat)
        var a = new OcrPoint(b.AxisX * b.HalfWidth / kx, b.AxisY * b.HalfWidth / ky);
        var v = new OcrPoint(-b.AxisY * b.HalfHeight / kx, b.AxisX * b.HalfHeight / ky);
        return new Quad(c - a - v, c + a - v, c + a + v, c - a + v);
    }

    public void Dispose() => _session.Dispose();
}
