namespace Tronnx.Ocr;

/// <summary>Egy pont képpixel-koordinátákban (a pixel bal felső sarka egész szám, a közepe +0,5).</summary>
public readonly record struct OcrPoint(float X, float Y)
{
    public static OcrPoint operator +(OcrPoint a, OcrPoint b) => new(a.X + b.X, a.Y + b.Y);
    public static OcrPoint operator -(OcrPoint a, OcrPoint b) => new(a.X - b.X, a.Y - b.Y);
    public static OcrPoint operator *(OcrPoint a, float k) => new(a.X * k, a.Y * k);
    public float Length => MathF.Sqrt(X * X + Y * Y);
}

/// <summary>
/// Négyszög (általában forgatott téglalap) a szöveg körül. A pontok sorrendje az olvasási
/// irányhoz igazodik: TopLeft → TopRight a szöveg alapvonalával párhuzamos.
/// </summary>
public readonly record struct Quad(OcrPoint TopLeft, OcrPoint TopRight, OcrPoint BottomRight, OcrPoint BottomLeft)
{
    public float Left => MathF.Min(MathF.Min(TopLeft.X, TopRight.X), MathF.Min(BottomRight.X, BottomLeft.X));
    public float Right => MathF.Max(MathF.Max(TopLeft.X, TopRight.X), MathF.Max(BottomRight.X, BottomLeft.X));
    public float Top => MathF.Min(MathF.Min(TopLeft.Y, TopRight.Y), MathF.Min(BottomRight.Y, BottomLeft.Y));
    public float Bottom => MathF.Max(MathF.Max(TopLeft.Y, TopRight.Y), MathF.Max(BottomRight.Y, BottomLeft.Y));

    /// <summary>A szöveg irányába eső oldal hossza.</summary>
    public float Width => (TopRight - TopLeft).Length;

    /// <summary>A szövegre merőleges oldal hossza (≈ betűmagasság + margó).</summary>
    public float Height => (BottomLeft - TopLeft).Length;

    public OcrPoint Center => new(
        (TopLeft.X + TopRight.X + BottomRight.X + BottomLeft.X) * 0.25f,
        (TopLeft.Y + TopRight.Y + BottomRight.Y + BottomLeft.Y) * 0.25f);

    /// <summary>A szöveg dőlésszöge fokban (0 = vízszintes).</summary>
    public float AngleDegrees => MathF.Atan2(TopRight.Y - TopLeft.Y, TopRight.X - TopLeft.X) * 180f / MathF.PI;

    /// <summary>
    /// A négyszög egy szelete a szöveg iránya mentén: <paramref name="from"/> és <paramref name="to"/>
    /// 0..1 közötti arányok. Ebből számoljuk a szavak dobozát a sor dobozából.
    /// </summary>
    public Quad SliceAlongWidth(float from, float to)
    {
        from = Math.Clamp(from, 0f, 1f);
        to = Math.Clamp(to, from, 1f);
        var top = TopRight - TopLeft;
        var bottom = BottomRight - BottomLeft;
        return new Quad(
            TopLeft + top * from,
            TopLeft + top * to,
            BottomLeft + bottom * to,
            BottomLeft + bottom * from);
    }

    /// <summary>A pontokat a [0, width] × [0, height] tartományba szorítja.</summary>
    public Quad ClampTo(int width, int height)
    {
        OcrPoint C(OcrPoint p) => new(Math.Clamp(p.X, 0f, width), Math.Clamp(p.Y, 0f, height));
        return new Quad(C(TopLeft), C(TopRight), C(BottomRight), C(BottomLeft));
    }
}
