using SkiaSharp;

namespace Tronnx.Ocr.Imaging;

/// <summary>
/// Képbetöltés SkiaSharp-pal (PNG, JPEG, BMP, GIF, WebP, …), EXIF-tájolás kezelésével.
/// A Skia csak a dekódolásra kell; utána minden a saját <see cref="BgrImage"/>-en fut.
/// </summary>
public static class ImageLoader
{
    public static BgrImage Load(string path)
    {
        using var stream = File.OpenRead(path);
        return Load(stream);
    }

    public static BgrImage Load(Stream stream)
    {
        // Az SKCodec visszatekerhető streamet szeret; hálózati / tömörített streamet bemásolunk.
        Stream input = stream;
        MemoryStream? copy = null;
        if (!stream.CanSeek)
        {
            copy = new MemoryStream();
            stream.CopyTo(copy);
            copy.Position = 0;
            input = copy;
        }

        try
        {
            // SKData-n keresztül, mert az SKCodec.Create(Stream) a hívó streamjét is lezárná.
            using var data = SKData.Create(input)
                ?? throw new InvalidDataException("Cannot read image stream.");
            using var codec = SKCodec.Create(data)
                ?? throw new InvalidDataException("Unsupported or corrupt image.");

            var info = new SKImageInfo(codec.Info.Width, codec.Info.Height, SKColorType.Bgra8888, SKAlphaType.Premul);
            using var bitmap = new SKBitmap(info);
            var result = codec.GetPixels(info, bitmap.GetPixels());
            if (result != SKCodecResult.Success && result != SKCodecResult.IncompleteInput)
                throw new InvalidDataException($"Image decoding failed: {result}.");

            var image = BgrImage.FromBgra(bitmap.GetPixelSpan(), info.Width, info.Height, bitmap.RowBytes, premultiplied: true);

            // A telefonos fotók gyakran elforgatva tárolódnak, a forgatás csak az EXIF-ben van.
            return codec.EncodedOrigin switch
            {
                SKEncodedOrigin.RightTop => image.RotateClockwise(1),
                SKEncodedOrigin.BottomRight => image.RotateClockwise(2),
                SKEncodedOrigin.LeftBottom => image.RotateClockwise(3),
                _ => image,
            };
        }
        finally
        {
            copy?.Dispose();
        }
    }

    /// <summary>Már memóriában lévő SKBitmap átalakítása (pl. PDF-oldal renderelése után).</summary>
    public static BgrImage FromBitmap(SKBitmap bitmap)
    {
        if (bitmap.ColorType == SKColorType.Bgra8888)
        {
            return BgrImage.FromBgra(bitmap.GetPixelSpan(), bitmap.Width, bitmap.Height, bitmap.RowBytes,
                premultiplied: bitmap.AlphaType == SKAlphaType.Premul);
        }

        using var converted = bitmap.Copy(SKColorType.Bgra8888)
            ?? throw new InvalidOperationException($"Cannot convert bitmap from {bitmap.ColorType}.");
        return BgrImage.FromBgra(converted.GetPixelSpan(), converted.Width, converted.Height, converted.RowBytes,
            premultiplied: converted.AlphaType == SKAlphaType.Premul);
    }
}
