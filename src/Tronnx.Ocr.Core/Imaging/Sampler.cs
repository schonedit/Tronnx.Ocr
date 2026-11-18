namespace Tronnx.Ocr.Imaging;

/// <summary>Csatornánkénti normalizálás: <c>érték = bájt * Scale[c] + Bias[c]</c> (c = B, G, R).</summary>
internal readonly struct ChannelNorm
{
    public ChannelNorm(float[] scale, float[] bias)
    {
        Scale = scale;
        Bias = bias;
    }

    public float[] Scale { get; }
    public float[] Bias { get; }

    /// <summary>ImageNet-átlag/szórás – a DB detektor bemenete.</summary>
    public static ChannelNorm ImageNet { get; } = FromMeanStd(
        mean: [0.485f, 0.456f, 0.406f],
        std: [0.229f, 0.224f, 0.225f]);

    /// <summary>(x/255 - 0,5) / 0,5 → [-1, 1] – a felismerő bemenete.</summary>
    public static ChannelNorm MinusOneToOne { get; } = FromMeanStd(
        mean: [0.5f, 0.5f, 0.5f],
        std: [0.5f, 0.5f, 0.5f]);

    private static ChannelNorm FromMeanStd(float[] mean, float[] std)
    {
        var scale = new float[3];
        var bias = new float[3];
        for (int c = 0; c < 3; c++)
        {
            scale[c] = 1f / (255f * std[c]);
            bias[c] = -mean[c] / std[c];
        }
        return new ChannelNorm(scale, bias);
    }
}

/// <summary>
/// Az egész csővezeték egyetlen képfeldolgozó primitíve: egy (akár elforgatott) téglalapot
/// mintavételez a forrásképből, és közvetlenül a modell CHW float bemenetébe írja, normalizálva.
///
/// Miért jó ez? Az 1.x-ben a kivágás → átméretezés → normalizálás → tenzorba másolás négy
/// külön lépés volt, mindegyik saját köztes képpel. Itt egyetlen menet, köztes puffer nélkül.
/// </summary>
internal static class Sampler
{
    /// <summary>
    /// A cél (x, y) pixel közepe a forrásban: <c>origin + (x+0,5)·du + (y+0,5)·dv</c>.
    /// Ha egy célpixel több forráspixelt fed le (kicsinyítés), több almintát átlagolunk (max. 4×4),
    /// hogy a vékony betűvonalak ne „szakadjanak szét" (aliasing).
    /// </summary>
    /// <param name="tensor">A cél tenzor ezen elemre eső része (C×H×W, a 0. csatorna elejétől).</param>
    /// <param name="tensorWidth">A tenzor sorhossza (kötegben ez a legszélesebb elem szélessége).</param>
    /// <param name="planeSize">Egy csatorna mérete a tenzorban (tensorHeight × tensorWidth).</param>
    public static void SampleToChw(
        BgrImage src, OcrPoint origin, OcrPoint du, OcrPoint dv,
        int dstWidth, int dstHeight,
        Span<float> tensor, int tensorWidth, int planeSize, in ChannelNorm norm)
    {
        float scaleU = du.Length, scaleV = dv.Length;
        int n = Math.Clamp((int)MathF.Ceiling(MathF.Max(scaleU, scaleV)), 1, 4);
        float inv = 1f / (n * n);

        var px = src.Pixels;
        int w = src.Width, h = src.Height, stride = w * 3;
        float s0 = norm.Scale[0], s1 = norm.Scale[1], s2 = norm.Scale[2];
        float b0 = norm.Bias[0], b1 = norm.Bias[1], b2 = norm.Bias[2];

        for (int y = 0; y < dstHeight; y++)
        {
            int row = y * tensorWidth;
            for (int x = 0; x < dstWidth; x++)
            {
                float sumB = 0, sumG = 0, sumR = 0;
                for (int j = 0; j < n; j++)
                {
                    float fy = y + (j + 0.5f) / n;
                    for (int i = 0; i < n; i++)
                    {
                        float fx = x + (i + 0.5f) / n;
                        // Folytonos forráskoordináta → pixelindex-tér (pixelközép = egész).
                        float sx = origin.X + fx * du.X + fy * dv.X - 0.5f;
                        float sy = origin.Y + fx * du.Y + fy * dv.Y - 0.5f;

                        int x0 = (int)MathF.Floor(sx), y0 = (int)MathF.Floor(sy);
                        float ax = sx - x0, ay = sy - y0;
                        int x1 = x0 + 1, y1 = y0 + 1;
                        // Szélen a szélső pixelt ismételjük (clamp).
                        x0 = x0 < 0 ? 0 : (x0 >= w ? w - 1 : x0);
                        x1 = x1 < 0 ? 0 : (x1 >= w ? w - 1 : x1);
                        y0 = y0 < 0 ? 0 : (y0 >= h ? h - 1 : y0);
                        y1 = y1 < 0 ? 0 : (y1 >= h ? h - 1 : y1);

                        int p00 = y0 * stride + x0 * 3, p01 = y0 * stride + x1 * 3;
                        int p10 = y1 * stride + x0 * 3, p11 = y1 * stride + x1 * 3;
                        float w00 = (1 - ax) * (1 - ay), w01 = ax * (1 - ay), w10 = (1 - ax) * ay, w11 = ax * ay;

                        sumB += px[p00] * w00 + px[p01] * w01 + px[p10] * w10 + px[p11] * w11;
                        sumG += px[p00 + 1] * w00 + px[p01 + 1] * w01 + px[p10 + 1] * w10 + px[p11 + 1] * w11;
                        sumR += px[p00 + 2] * w00 + px[p01 + 2] * w01 + px[p10 + 2] * w10 + px[p11 + 2] * w11;
                    }
                }

                int o = row + x;
                tensor[o] = sumB * inv * s0 + b0;
                tensor[planeSize + o] = sumG * inv * s1 + b1;
                tensor[2 * planeSize + o] = sumR * inv * s2 + b2;
            }
        }
    }
}
