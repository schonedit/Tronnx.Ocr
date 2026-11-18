namespace Tronnx.Ocr.Imaging;

/// <summary>
/// Egyszerű, 8 bites BGR kép (sorfolytonos, pixelenként 3 bájt). Szándékosan nem SKBitmap:
/// így a mintavételező saját, tömör ciklusban olvashat belőle, natív hívások nélkül.
/// A BGR sorrend azért kell, mert a PaddleOCR-modelleket OpenCV-vel (BGR) tanították.
/// </summary>
public sealed class BgrImage
{
    public BgrImage(int width, int height, byte[] pixels)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), "Image size must be positive.");
        if (pixels.Length < width * height * 3)
            throw new ArgumentException("Pixel buffer is too small.", nameof(pixels));
        Width = width;
        Height = height;
        Pixels = pixels;
    }

    public int Width { get; }
    public int Height { get; }

    /// <summary>B, G, R bájtok, soronként <c>Width * 3</c> bájt, sorközi kitöltés nélkül.</summary>
    public byte[] Pixels { get; }

    /// <summary>
    /// BGRA (Skia <c>Bgra8888</c>) pufferből készít BGR képet. Az átlátszó pixeleket fehér
    /// háttérre keveri – különben egy átlátszó PNG fekete szövege fekete háttéren tűnne el.
    /// </summary>
    public static BgrImage FromBgra(ReadOnlySpan<byte> bgra, int width, int height, int rowBytes, bool premultiplied)
    {
        var dst = new byte[width * height * 3];
        for (int y = 0; y < height; y++)
        {
            var row = bgra.Slice(y * rowBytes, width * 4);
            int o = y * width * 3;
            for (int x = 0; x < width; x++)
            {
                int i = x * 4;
                int a = row[i + 3];
                if (a == 255)
                {
                    dst[o++] = row[i];
                    dst[o++] = row[i + 1];
                    dst[o++] = row[i + 2];
                }
                else if (premultiplied)
                {
                    // premultiplied: c' = c*a/255, fehér háttérrel: c' + (255 - a)
                    int bg = 255 - a;
                    dst[o++] = (byte)Math.Min(255, row[i] + bg);
                    dst[o++] = (byte)Math.Min(255, row[i + 1] + bg);
                    dst[o++] = (byte)Math.Min(255, row[i + 2] + bg);
                }
                else
                {
                    int bg = (255 - a) * 255;
                    dst[o++] = (byte)((row[i] * a + bg) / 255);
                    dst[o++] = (byte)((row[i + 1] * a + bg) / 255);
                    dst[o++] = (byte)((row[i + 2] * a + bg) / 255);
                }
            }
        }
        return new BgrImage(width, height, dst);
    }

    /// <summary>Elforgatja a képet az óramutató járásával megegyezően, 90°-os lépésekben.</summary>
    public BgrImage RotateClockwise(int quarterTurns)
    {
        quarterTurns = ((quarterTurns % 4) + 4) % 4;
        if (quarterTurns == 0)
            return this;

        int w = Width, h = Height;
        int nw = quarterTurns == 2 ? w : h;
        int nh = quarterTurns == 2 ? h : w;
        var src = Pixels;
        var dst = new byte[nw * nh * 3];

        for (int y = 0; y < nh; y++)
        {
            for (int x = 0; x < nw; x++)
            {
                // Melyik forráspixel kerül az (x, y) helyre?
                int sx, sy;
                switch (quarterTurns)
                {
                    case 1: sx = y; sy = h - 1 - x; break;          // 90° jobbra
                    case 2: sx = w - 1 - x; sy = h - 1 - y; break;  // 180°
                    default: sx = w - 1 - y; sy = x; break;         // 270° jobbra
                }
                int s = (sy * w + sx) * 3, d = (y * nw + x) * 3;
                dst[d] = src[s];
                dst[d + 1] = src[s + 1];
                dst[d + 2] = src[s + 2];
            }
        }
        return new BgrImage(nw, nh, dst);
    }
}
