namespace Tronnx.Ocr;

/// <summary>Egy felismert szó. A doboza a sor dobozából, a CTC-időlépések alapján becsült.</summary>
public sealed record OcrWord(string Text, float Confidence, Quad Box);

/// <summary>Egy felismert szövegsor (a detektor egy doboza).</summary>
public sealed record OcrLine(string Text, float Confidence, Quad Box, IReadOnlyList<OcrWord> Words);

/// <summary>Az egyes lépések ideje ezredmásodpercben – mérésekhez és optimalizáláshoz.</summary>
public sealed record OcrTimings(double DetectionMs, double RecognitionMs, double TotalMs);

/// <summary>Egy kép (oldal) teljes OCR-eredménye, olvasási sorrendben.</summary>
public sealed class OcrPage
{
    public OcrPage(int width, int height, IReadOnlyList<OcrLine> lines, OcrTimings timings)
    {
        Width = width;
        Height = height;
        Lines = lines;
        Timings = timings;
        Text = BuildText(lines);
    }

    /// <summary>A feldolgozott kép szélessége pixelben (EXIF-forgatás után).</summary>
    public int Width { get; }

    /// <summary>A feldolgozott kép magassága pixelben (EXIF-forgatás után).</summary>
    public int Height { get; }

    public IReadOnlyList<OcrLine> Lines { get; }

    public OcrTimings Timings { get; }

    /// <summary>A teljes szöveg: egy fizikai sorban lévő dobozok szóközzel, a sorok sortöréssel elválasztva.</summary>
    public string Text { get; }

    public IEnumerable<OcrWord> Words => Lines.SelectMany(l => l.Words);

    private static string BuildText(IReadOnlyList<OcrLine> lines)
    {
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < lines.Count; i++)
        {
            if (i > 0)
                sb.Append(ReadingOrder.SameRow(lines[i - 1].Box, lines[i].Box) ? ' ' : '\n');
            sb.Append(lines[i].Text);
        }
        return sb.ToString();
    }
}
