using OpenCvSharp;
using PdfSharpCore.Drawing;
using PdfSharpCore.Pdf;
using SkiaSharp;
using Tronnx.Ocr.Types;

namespace Tronnx.Ocr
{
    internal static class SearchablePdf
    {
        /// <summary>
        /// Recompresses a JPEG image to a specified quality level using SkiaSharp.
        /// </summary>
        /// <param name="inputPath"></param>
        /// <param name="quality"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        /// <exception cref="FileNotFoundException"></exception>
        /// <exception cref="InvalidOperationException"></exception>
        private static string RecompressJpeg(string inputPath, int quality = 70)
        {
            if (string.IsNullOrEmpty(inputPath))
                throw new ArgumentException("Input path cannot be null or empty.", nameof(inputPath));
            if (!File.Exists(inputPath))
                throw new FileNotFoundException("Input file not found.", inputPath);

            using var input = File.OpenRead(inputPath);
            var bitmap = SKBitmap.Decode(input);
            if (bitmap == null)
                throw new InvalidOperationException("Failed to decode image with SkiaSharp.");

            using (bitmap)
            {
                using var data = bitmap.Encode(SKEncodedImageFormat.Jpeg, quality);
                if (data == null)
                    throw new InvalidOperationException("Failed to encode JPEG with SkiaSharp.");

                string output = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".jpg");
                using var outStream = File.OpenWrite(output);
                data.SaveTo(outStream);
                return output;
            }
        }

        /// <summary>
        /// Creates a searchable PDF by drawing each source image as a page background and overlaying invisible text
        /// annotations for recognized text boxes. Each <see cref="PageResult"/> becomes one PDF page using the
        /// page image as the visible layer; recognized text is drawn transparent so the PDF becomes searchable
        /// while remaining visually identical to the input image.
        /// </summary>
        /// <param name="outputPath">Full path where the PDF will be written. The method will overwrite existing files.</param>
        /// <param name="pages">List of <see cref="PageResult"/> describing each page image and detected text boxes.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="pages"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when <paramref name="outputPath"/> is null or empty.</exception>
        /// <exception cref="IOException">Thrown for disk I/O errors while reading/writing images or saving the PDF.</exception>
        internal static string GenerateSearchablePdf(string outputPath, List<PageResult> pages)
        {
            if (string.IsNullOrEmpty(outputPath))
                throw new ArgumentException("Output path cannot be null or empty.", nameof(outputPath));
            if (pages is null)
                throw new ArgumentException(nameof(pages));

            using var document = new PdfDocument();
            var tempFiles = new List<string>();
            string text = "";
            try
            {
                foreach (var pageResult in pages)
                {
                    var page = document.AddPage();
                    page.Width = pageResult.ImageWidth;
                    page.Height = pageResult.ImageHeight;

                    using var gfx = XGraphics.FromPdfPage(page);
                    var compressedPath = RecompressJpeg(pageResult.ImagePath, 70);
                    tempFiles.Add(compressedPath);

                    using var img = XImage.FromFile(compressedPath);
                    gfx.DrawImage(img, 0, 0, page.Width, page.Height);

                    foreach (var box in pageResult.Boxes)
                    {
                        if (string.IsNullOrWhiteSpace(box.Text))
                            continue;

                        double x = box.X;
                        double y = box.Y;
                        double w = box.Width;
                        double h = box.Height;

                        if (w <= 0 || h <= 0)
                            continue;

                        double baseFontSize = h * 0.7;

                        if (baseFontSize < 4)
                            baseFontSize = 4;

                        var font = new XFont("Arial", baseFontSize, XFontStyle.Regular);
                        var transparentBrush = new XSolidBrush(XColor.FromArgb(0, 0, 0, 0));

                        XSize natural = gfx.MeasureString(box.Text, font);
                        if (natural.Width <= 0)
                            continue;

                        const double Fudge = 1.05;
                        double scaleX = (w * Fudge) / natural.Width;

                        text += box.Text + " ";
                        gfx.Save();
                        gfx.TranslateTransform(x, y);
                        gfx.ScaleTransform(scaleX, 1.0);
                        gfx.DrawLine(XPens.Red, 0, 0, 0, 0);
                        gfx.DrawString(box.Text + " ", font, transparentBrush, new XPoint(0, 0), XStringFormats.TopLeft);
                        gfx.Restore();
                    }
                }
                document.Save(outputPath);
                return text;
            }
            finally
            {
                foreach (var f in tempFiles)
                {
                    try { if (File.Exists(f)) File.Delete(f); } catch { /* ignore cleanup errors */ }
                }
            }
        }

        /// <summary>
        /// Processes a single page image: runs detection, remaps rotated boxes to the original image,
        /// performs per-box recognition and returns a <see cref="PageResult"/> containing the image
        /// metadata and axis-aligned detected boxes. Boxes are clamped to the image bounds and
        /// recognized text is trimmed and sanitized.
        /// </summary>
        /// <param name="page">The OpenCvSharp <see cref="Mat"/> containing the page image. Must be non-null and non-empty.</param>
        /// <param name="imagePath">The original image path (used for result metadata and error context).</param>
        /// <returns>A <see cref="PageResult"/> with image dimensions and detected text boxes.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="page"/> or <paramref name="imagePath"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when <paramref name="page"/> is empty or <paramref name="imagePath"/> is empty.</exception>
        /// <exception cref="InvalidOperationException">Thrown when internal detection/recognition fails; inner exception preserved for diagnostics.</exception>
        internal static PageResult ProcessPageToResult(Detector detector, Recognizer recognizer, Mat page, string imagePath)
        {
            if (page is null)
                throw new ArgumentNullException(nameof(page));
            if (string.IsNullOrEmpty(imagePath))
                throw new ArgumentException("Image path cannot be null or empty.", nameof(imagePath));
            if (page.Empty())
                throw new ArgumentException("Input image cannot be empty.", nameof(page));
            try
            {
                var detInput = detector.PreprocessForDetector(page, 1024, out float scale, out int padX, out int padY);
                var detOutput = detector.RunDetector(detInput);
                var rotatedBoxes = detector.PostprocessBoxes(detOutput, 0.001f, 50) ?? new List<OpenCvSharp.RotatedRect>();
                rotatedBoxes = detector.RemapBoxesToOriginal(rotatedBoxes, scale, padX, padY);
                detector.OrderReading(rotatedBoxes);

                var texts = recognizer.RecognizePerBox(page, rotatedBoxes, 0, targetH: 32) ?? new List<string>();
                var detectedBoxes = new List<DetectedBox>();

                for (int i = 0; i < rotatedBoxes.Count; i++)
                {
                    var rr = rotatedBoxes[i];

                    if (rr.Center.X < 0 && rr.Center.Y < 0)
                        continue;

                    var text = (i < texts.Count) ? texts[i] : null;
                    if (string.IsNullOrWhiteSpace(text))
                        continue;

                    var rect = rr.BoundingRect();

                    int x = Math.Max(0, rect.X);
                    int y = Math.Max(0, rect.Y);
                    int w = rect.Width;
                    int h = rect.Height;

                    if (w <= 0 || h <= 0)
                        continue;

                    if (x + w > page.Width)
                        w = page.Width - x;
                    if (y + h > page.Height)
                        h = page.Height - y;

                    if (w <= 0 || h <= 0)
                        continue;

                    detectedBoxes.Add(new DetectedBox
                    {
                        X = x,
                        Y = y,
                        Width = w,
                        Height = h,
                        Text = text
                    });
                }

                return new PageResult
                {
                    ImagePath = imagePath,
                    ImageWidth = page.Width,
                    ImageHeight = page.Height,
                    Boxes = detectedBoxes
                };
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to process page '{imagePath}'. See inner exception for details.", ex);
            }
        }

        /// <summary>
        /// Generates a searchable PDF document from OCR-processed page resultOptionally highlights specified target words by drawing 
        /// a semi-transparent rectangle behind matching text boxes.
        /// </summary>
        /// <param name="outputPath">The full file path where the generated PDF will be saved. Existing files will be overwritten.</param>
        /// <param name="pages">The collection of <see cref="PageResult"/> objects containing the page images and their recognized 
        /// text boxes with coordinates.</param>
        /// <param name="targets">A list of words or phrases to highlight. Matching text boxes will be visually emphasized in the final PDF.</param>
        /// <returns>The concatenated recognized text from all pages.</returns>
        /// <exception cref="ArgumentException"></exception>
        internal static string HighlightPdf(string outputPath, List<PageResult> pages, IEnumerable<string> targets)
        {
            if (string.IsNullOrEmpty(outputPath))
                throw new ArgumentException("Output path cannot be null or empty.", nameof(outputPath));
            if (pages is null)
                throw new ArgumentException(nameof(pages));
            if (targets is null)
                throw new ArgumentException(nameof(targets));

            using var document = new PdfDocument();
            var tempFiles = new List<string>();
            string text = "";
            var highlightBrush = new XSolidBrush(XColor.FromArgb(100, 255, 255, 0));
            try
            {
                foreach (var pageResult in pages)
                {
                    var page = document.AddPage();
                    page.Width = pageResult.ImageWidth;
                    page.Height = pageResult.ImageHeight;

                    using var gfx = XGraphics.FromPdfPage(page);
                    var compressedPath = RecompressJpeg(pageResult.ImagePath, 70);
                    tempFiles.Add(compressedPath);

                    using var img = XImage.FromFile(compressedPath);
                    gfx.DrawImage(img, 0, 0, page.Width, page.Height);

                    foreach (var box in pageResult.Boxes)
                    {
                        if (string.IsNullOrWhiteSpace(box.Text))
                            continue;

                        double x = box.X;
                        double y = box.Y;
                        double w = box.Width;
                        double h = box.Height;

                        if (w <= 0 || h <= 0)
                            continue;

                        double baseFontSize = h * 0.7;

                        if (baseFontSize < 4)
                            baseFontSize = 4;

                        var font = new XFont("Arial", baseFontSize, XFontStyle.Regular);
                        var transparentBrush = new XSolidBrush(XColor.FromArgb(0, 0, 0, 0));

                        XSize natural = gfx.MeasureString(box.Text, font);
                        if (natural.Width <= 0)
                            continue;

                        const double Fudge = 1.05;
                        double scaleX = (w * Fudge) / natural.Width;

                        if (targets.Any(x => x.Contains(box.Text, StringComparison.OrdinalIgnoreCase)))
                        {
                            gfx.DrawRectangle(highlightBrush, x, y, w, h);
                        }

                        text += box.Text + " ";
                        gfx.Save();
                        gfx.TranslateTransform(x, y);
                        gfx.ScaleTransform(scaleX, 1.0);
                        gfx.DrawLine(XPens.Red, 0, 0, 0, 0);
                        gfx.DrawString(box.Text + " ", font, transparentBrush, new XPoint(0, 0), XStringFormats.TopLeft);
                        gfx.Restore();
                    }
                }
                document.Save(outputPath);
                return text;
            }
            finally
            {
                foreach (var f in tempFiles)
                {
                    try { if (File.Exists(f)) File.Delete(f); } catch { /* ignore cleanup errors */ }
                }
            }
        }
    }
}
