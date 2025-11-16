using OpenCvSharp;
using System.Text;

namespace Tronnx.Ocr
{
    /// <summary>
    /// Provides a high-level OCR pipeline using ONNX models for text detection and recognition.
    /// This class loads the detector and recognizer models, validates input images,
    /// and exposes convenience methods for extracting text from image files.
    /// Intended for use inside a NuGet package as a drop-in OCR utility.    
    /// /// </summary>
    public class TronnxOCR : IDisposable
    {
        private static readonly string[] AllowedExtensions = { ".png", ".jpg", ".jpeg", ".bmp", ".tiff" };
        private Detector detector { get; }
        private Recognizer recognizer { get; }

        public TronnxOCR()
        {
            string detPath = ExtractModelAndSidecar("linknet_resnet18.onnx");
            string recPath = ExtractModelAndSidecar("crnn_vgg16_bn_dynW.onnx");

            string vocab = ExtractEmbeddedText("vocab_crnn.txt");
            string blank = ExtractEmbeddedText("blank_crnn.txt");

            //string vocab = File.ReadAllText(vocabPath);
            int blankIndex = int.TryParse(blank, out var b) ? b : vocab.Length;

            detector = new Detector(detPath);
            recognizer = new Recognizer(recPath, vocab, blankIndex);
        }

        /// <summary>
        /// Processes an image file and returns all recognized text extracted from it.
        /// The input must be an image file (e.g., PNG or JPEG). 
        /// For best results, ensure the image is properly oriented and not rotated or skewed.
        /// </summary>
        /// <param name="imagePath">
        /// The full file path of the image to be processed. The file must exist and be a supported image format.
        /// </param>
        /// <returns>
        /// A string containing the recognized text extracted from the image.
        /// </returns>
        public string GetText(string imagePath)
        {
            var ext = Path.GetExtension(imagePath)?.ToLowerInvariant();

            if (string.IsNullOrEmpty(ext) || !AllowedExtensions.Contains(ext))
                throw new NotSupportedException($"Unsupported file type: '{ext}'. Only image files are allowed: {string.Join(", ", AllowedExtensions)}");

            var img = PreProcessImage(imagePath);
            return ProcessImage(img);
        }

        /// <summary>
        /// Loads and preprocesses an image from the specified file path.
        /// </summary>
        /// <param name="imagePath">The path to the image file to be loaded. The file must exist and be in a supported format.</param>
        /// <returns>A <see cref="Mat"/> object representing the loaded image.</returns>
        /// <exception cref="InvalidOperationException">Thrown if the image cannot be loaded, which may occur if the file is missing, unreadable, or in an
        /// unsupported format.</exception>
        private Mat PreProcessImage(string imagePath)
        {
            var img = Cv2.ImRead(imagePath);

            if (img.Empty())
                throw new InvalidOperationException("Failed to load the image. The file may be missing, unreadable, or unsupported.");

            return img;
        }

        /// <summary>
        /// Performs the full OCR pipeline on an already loaded image.
        /// This includes detection, box postprocessing, ordering, and text recognition.
        /// </summary>
        /// <param name="page">
        /// The <see cref="Mat"/> image to process. Must contain readable text regions
        /// for the OCR pipeline to detect and recognize.
        /// </param>
        /// <returns>
        /// A string containing all recognized text, ordered in reading order.
        /// </returns>
        private string ProcessImage(Mat page)
        {
            // 1) preprocess
            var detInput = detector.PreprocessForDetector(page, 1024, out float scale, out int padX, out int padY);

            // 2) detect
            var detOutput = detector.RunDetector(detInput);

            // 3) boxes
            var boxes = detector.PostprocessBoxes(detOutput, 0.001f, 50);
            boxes = detector.RemapBoxesToOriginal(boxes, scale, padX, padY);
            detector.OrderReading(boxes);

            // 4) recognize 
            var lines = recognizer.RecognizeAll(page, boxes, 0, targetH: 32);
            return string.Join(Environment.NewLine, lines);
        }

        /// <summary>
        /// Reads an embedded text resource (e.g., vocab or blank file) from the assembly and returns its content.
        /// Normalizes line endings to LF (<c>\n</c>) for consistent cross-platform behavior.
        /// </summary>
        /// <param name="fileName">
        /// The file name (not the full resource name). The method will search for any embedded resource
        /// whose name ends with this file name.
        /// </param>
        /// <returns>The full textual content of the embedded file.</returns>
        /// <exception cref="FileNotFoundException">Thrown if the resource is not found in the assembly.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the resource stream cannot be opened.</exception>

        private string ExtractEmbeddedText(string fileName)
        {
            var asm = typeof(TronnxOCR).Assembly;
            var resourceName = asm.GetManifestResourceNames().FirstOrDefault(n => n.EndsWith(fileName, StringComparison.OrdinalIgnoreCase));

            if (resourceName == null)
                throw new FileNotFoundException($"Embedded text resource not found: {fileName}");

            using var stream = asm.GetManifestResourceStream(resourceName);
            if (stream == null)
                throw new InvalidOperationException($"Failed to load embedded resource: {resourceName}");

            using var reader = new StreamReader(stream, Encoding.UTF8);
            var text = reader.ReadToEnd();
            return text.Replace("\r\n", "\n").Replace("\r", "\n");
        }

        /// <summary>
        /// Ensures that both the main ONNX model file and its optional <c>.onnx.data</c> sidecar file
        /// are available in the temporary cache directory.  
        /// If the files already exist in the cache, they will not be copied again.
        /// </summary>
        /// <param name="modelFile">The ONNX model file name (e.g. <c>linknet_resnet18.onnx</c>).</param>
        /// <returns>
        /// The full path to the cached primary model file.  
        /// (The sidecar, if present, will be cached with the same name in the same directory.)
        /// </returns>
        private string ExtractModelAndSidecar(string modelFile)
        {
            string baseName = Path.GetFileName(modelFile);
            string sideName = baseName + ".onnx.data";
            // Cache or reuse the primary ONNX model file
            string modelPath = ExtractCached(baseName);

            // Try caching the sidecar file — ignore if missing
            try
            {
                ExtractCached(sideName);
            }
            catch
            {
                // Optional file — intentionally ignored
            }

            return modelPath;
        }

        /// <summary>
        /// Extracts an embedded resource to a well-known cache directory inside the OS temp folder.
        /// If the file already exists, no extraction is performed.
        /// </summary>
        /// <param name="fileName">
        /// The name of the embedded resource file (e.g. <c>linknet_resnet18.onnx</c>).  
        /// The method will search for any embedded resource whose name ends with this file name.
        /// </param>
        /// <returns>The full path to the cached file.</returns>
        /// <exception cref="FileNotFoundException">Thrown if the resource cannot be found in the assembly.</exception>

        private string ExtractCached(string fileName)
        {
            string cacheDir = Path.Combine(Path.GetTempPath(), "pp_onnx_cache");
            Directory.CreateDirectory(cacheDir);

            string target = Path.Combine(cacheDir, fileName);

            // If already cached, no need to copy again
            if (!File.Exists(target))
            {
                ExtractResourceToSpecificPath(fileName, target);
            }

            return target;
        }

        /// <summary>
        /// Extracts an embedded binary resource to a specific output path.
        /// This method always overwrites the output file if it exists.
        /// </summary>
        /// <param name="resourceName">
        /// The file name of the embedded resource to extract (suffix match is used).
        /// </param>
        /// <param name="outputPath">The full target file path on disk.</param>
        /// <exception cref="FileNotFoundException">Thrown if the resource cannot be found.</exception>
        /// <exception cref="IOException">
        /// Thrown if the output file cannot be created (e.g. permission issues).
        /// </exception>
        private void ExtractResourceToSpecificPath(string resourceName, string outputPath)
        {
            var asm = typeof(TronnxOCR).Assembly;

            var resource = asm.GetManifestResourceNames()
                .FirstOrDefault(n => n.EndsWith(resourceName, StringComparison.OrdinalIgnoreCase));

            if (resource == null)
                throw new FileNotFoundException($"Embedded resource not found: {resourceName}");

            using var stream = asm.GetManifestResourceStream(resource);
            using var fs = File.Create(outputPath);

            stream.CopyTo(fs);
        }

        public void Dispose()
        {
            detector?.Dispose();
            recognizer?.Dispose();
        }
    }
}
