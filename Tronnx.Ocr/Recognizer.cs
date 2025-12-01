using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Text;
using Tronnx.Ocr.Types;

namespace Tronnx.Ocr
{
    internal class Recognizer : IDisposable
    {
        static readonly float[] MEAN = { 0.694f, 0.695f, 0.693f };
        static readonly float[] STD = { 0.299f, 0.296f, 0.301f };

        readonly InferenceSession _session;
        readonly string _vocab;
        readonly int _blankIndex;

        internal Recognizer(string modelPath, string vocab, int blankIndex)
        {
            _vocab = vocab;
            _blankIndex = blankIndex;
            var so = new SessionOptions();
            so.AppendExecutionProvider_CPU();
            so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            _session = new InferenceSession(modelPath, so);
        }

        /// <summary>
        /// Recognizes text from a series of rotated rectangles within an image and returns the recognized text lines.
        /// </summary>
        /// <remarks>The method processes each rotated rectangle to extract and recognize text, merging text into
        /// lines based on vertical proximity. Empty or invalid rectangles are skipped. The method assumes that the
        /// vocabulary and inference session are correctly configured for the text recognition task.</remarks>
        /// <param name="rec">The inference session used for text recognition.</param>
        /// <param name="vocab">The vocabulary string used by the recognizer to interpret text.</param>
        /// <param name="pageBgr">The image from which text is to be recognized, represented as a matrix.</param>
        /// <param name="boxes">A list of rotated rectangles that define the regions of interest in the image for text recognition.</param>
        /// <param name="saved">A counter for the number of saved debug crops, used for debugging purposes.</param>
        /// <param name="targetH">The target height for preprocessing the text regions. Defaults to 32.</param>
        /// <returns>A list of strings, where each string represents a line of recognized text.</returns>
        internal List<string> RecognizeAll(Mat pageBgr, List<RotatedRect> boxes, int saved, int targetH = 32)
        {
            var lines = new List<string>();
            if (boxes.Count == 0) return lines;

            float lastY = boxes[0].Center.Y;
            float lineMergeTol = MedianHeight(boxes) * 0.7f;
            var sbLine = new StringBuilder();
            float runningY = boxes[0].Center.Y;

            for (int i = 0; i < boxes.Count; i++)
            {
                var rr = boxes[i];
                string blankPath = "blank_crnn.txt";
                using var crop0 = CropAabb(pageBgr, rr, padXRatio: 0.05f, padYRatio: 0.30f);

                if (rr.Center.X < 0 && rr.Center.Y < 0)
                {
                    if (sbLine.Length > 0)
                    {
                        lines.Add(sbLine.ToString().Trim());
                        sbLine.Clear();
                    }
                    continue;
                }

                using var crop = TrimWhitespace(crop0, padXRatio: 0.15f, padYRatio: 0.25f);
                var ten = PreprocessWordDynamic(crop0, targetH: 32, maxW: 512);
                int blankIndex = File.Exists(blankPath) && int.TryParse(File.ReadAllText(blankPath), out var b) ? b : _vocab.Length;
                var text = RunRecognizerCtcSingle(_session, _vocab, blankIndex, ten);

                if (!string.IsNullOrWhiteSpace(text))
                {
                    sbLine.Append(text);
                    sbLine.Append(' ');
                }
            }
            if (sbLine.Length > 0)
                lines.Add(sbLine.ToString().Trim());

            return lines;
        }

        /// <summary>
        /// Crops a specified region from an image based on a rotated rectangle, with optional padding.
        /// </summary>
        /// <remarks>The method calculates a bounding rectangle around the rotated rectangle, applies the
        /// specified padding, and ensures the resulting rectangle is within the bounds of the source image.</remarks>
        /// <param name="bgr">The source image from which to crop the region.</param>
        /// <param name="rr">The rotated rectangle defining the region to crop.</param>
        /// <param name="padXRatio">The ratio of horizontal padding to apply to the cropped region. Default is 0.12.</param>
        /// <param name="padYRatio">The ratio of vertical padding to apply to the cropped region. Default is 0.30.</param>
        /// <returns>A new image containing the cropped region with the specified padding applied.</returns>
        private Mat CropAabb(Mat bgr, RotatedRect rr, float padXRatio, float padYRatio)
        {
            var rect = Cv2.BoundingRect(rr.Points());

            int padLeft = (int)Math.Round(rect.Width * padXRatio);
            int padRight = (int)Math.Round(rect.Width * padXRatio * 1.5);
            int padTop = (int)Math.Round(rect.Height * padYRatio);
            int padBot = (int)Math.Round(rect.Height * padYRatio);

            var r = new Rect(
                rect.X - padLeft,
                rect.Y - padTop,
                rect.Width + padLeft + padRight,
                rect.Height + padTop + padBot
            );

            r.X = Math.Max(0, r.X);
            r.Y = Math.Max(0, r.Y);
            r.Width = Math.Min(bgr.Cols - r.X, r.Width);
            r.Height = Math.Min(bgr.Rows - r.Y, r.Height);

            return new Mat(bgr, r).Clone();
        }

        /// <summary>
        /// Preprocesses an image of a word by resizing and normalizing it into a tensor format suitable for model input.
        /// </summary>
        /// <remarks>The method ensures that the width of the resized image is even for stability. The image is
        /// converted from BGR to RGB color space before normalization.</remarks>
        /// <param name="cropBgr">The input image in BGR color space representing the cropped word.</param>
        /// <param name="targetH">The target height for the resized image. Defaults to 32.</param>
        /// <param name="maxW">The maximum allowable width for the resized image. Defaults to 384.</param>
        /// <returns>A <see cref="DenseTensor{T}"/> of floats with dimensions [1, 3, targetH, newW], where the image is resized to
        /// the specified height and a width that maintains the aspect ratio, clamped between 48 and <paramref
        /// name="maxW"/>. The tensor is normalized using predefined mean and standard deviation values.</returns>
        private DenseTensor<float> PreprocessWordDynamic(Mat cropBgr, int targetH = 32, int maxW = 384)
        {
            int w = cropBgr.Cols, h = cropBgr.Rows;
            int newW = (int)Math.Round((float)w / Math.Max(1, h) * targetH);
            newW = Math.Clamp(newW, 48, maxW);
            if ((newW & 1) == 1) newW++;

            using var resized = new Mat();
            Cv2.Resize(cropBgr, resized, new Size(newW, targetH));
            using var rgb = new Mat();
            Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);

            var t = new DenseTensor<float>(new[] { 1, 3, targetH, newW });
            var idx = rgb.GetGenericIndexer<Vec3b>();
            for (int y = 0; y < targetH; y++)
            {
                for (int x = 0; x < newW; x++)
                {
                    var p = idx[y, x];
                    float r = (p.Item0 / 255f - MEAN[0]) / STD[0];
                    float g = (p.Item1 / 255f - MEAN[1]) / STD[1];
                    float b = (p.Item2 / 255f - MEAN[2]) / STD[2];
                    t[0, 0, y, x] = r; t[0, 1, y, x] = g; t[0, 2, y, x] = b;
                }
            }
            return t;
        }

        /// <summary>
        /// Performs a greedy decoding of the given logits tensor using the specified vocabulary and blank index.
        /// </summary>
        /// <remarks>The method assumes that the logits tensor is either in [1, T, C] or [1, C, T] format, where T
        /// is the number of time steps and C is the number of classes. The method performs a greedy search to find the most
        /// likely character at each time step, appending it to the result if it is not a blank and not a repetition of the
        /// previous character.</remarks>
        /// <param name="logits">A tensor of shape [1, T, C] or [1, C, T] representing the logits for each time step and class.</param>
        /// <param name="vocab">A string representing the vocabulary, where each character corresponds to a class index.</param>
        /// <param name="blank">The index in the vocabulary that represents the blank token.</param>
        /// <returns>A string representing the decoded sequence, excluding repeated characters and blanks.</returns>
        private string CtcGreedyDecode(Tensor<float> logits, string vocab, int blankIndex)
        {
            // logits: [1, T, C] or [1, C, T]
            var dimensions = logits.Dimensions;

            bool isTimeMajor = dimensions[1] <= dimensions[2];
            int timeSteps = isTimeMajor ? dimensions[1] : dimensions[2];
            int numClasses = isTimeMajor ? dimensions[2] : dimensions[1];

            var decoded = new StringBuilder(timeSteps);
            int lastCharIndex = -1;

            for (int t = 0; t < timeSteps; t++)
            {
                int bestClassIndex = 0;
                float highestLogit = float.NegativeInfinity;

                for (int c = 0; c < numClasses; c++)
                {
                    float currentLogit = isTimeMajor ? logits[0, t, c] : logits[0, c, t];
                    if (currentLogit > highestLogit)
                    {
                        highestLogit = currentLogit;
                        bestClassIndex = c;
                    }
                }

                bool isBlank = (bestClassIndex == blankIndex);

                if (!isBlank && bestClassIndex == lastCharIndex)
                    continue;

                if ((uint)bestClassIndex < (uint)vocab.Length)
                {
                    var kimenet = vocab[bestClassIndex];
                    decoded.Append(vocab[bestClassIndex]);
                }

                lastCharIndex = bestClassIndex;
            }
            return decoded.ToString();
        }

        /// <summary>
        /// Calculates the median height of a collection of rotated rectangles.
        /// </summary>
        /// <param name="boxes">A list of <see cref="RotatedRect"/> objects from which to calculate the median height.</param>
        /// <returns>The median of the smaller dimension (height or width) of each rectangle in the list. Returns 32 if the list is
        /// empty.</returns>
        private float MedianHeight(List<RotatedRect> boxes)
        {
            var h = boxes.Select(b => Math.Min(b.Size.Height, b.Size.Width)).OrderBy(x => x).ToArray();
            if (h.Length == 0) return 32f;
            return h[h.Length / 2];
        }

        /// <summary>
        /// Trims the whitespace from the edges of an image and returns a new image with optional padding.
        /// </summary>
        /// <remarks>The method converts the source image to grayscale and applies a binary inverse threshold to
        /// identify the non-whitespace areas. It calculates the bounding box of the non-whitespace region and applies the
        /// specified padding ratios to determine the final cropped area.</remarks>
        /// <param name="src">The source image from which to trim whitespace.</param>
        /// <param name="padXRatio">The ratio of padding to add to the width of the trimmed image. Default is 0.08.</param>
        /// <param name="padYRatio">The ratio of padding to add to the height of the trimmed image. Default is 0.20.</param>
        /// <returns>A new image with whitespace trimmed and optional padding applied.</returns>
        private Mat TrimWhitespace(Mat src, float padXRatio = 0.08f, float padYRatio = 0.20f)
        {
            using var gray = new Mat();
            Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);

            using var bin = new Mat();
            Cv2.Threshold(gray, bin, 0, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);

            int W = bin.Cols, H = bin.Rows;

            using var colSum = new Mat();
            Cv2.Reduce(bin, colSum, 0, ReduceTypes.Sum, (int)MatType.CV_32SC1);

            using var rowSum = new Mat();
            Cv2.Reduce(bin, rowSum, (ReduceDimension)1, ReduceTypes.Sum, (int)MatType.CV_32SC1);

            int minInkCol = Math.Max(1, (int)(0.01 * 255 * H));
            int minInkRow = Math.Max(1, (int)(0.01 * 255 * W));

            int left = 0, right = W - 1, top = 0, bottom = H - 1;

            var cs = colSum.GetGenericIndexer<int>();
            while (left < W && cs[0, left] < minInkCol) left++;
            while (right > left && cs[0, right] < minInkCol) right--;

            var rs = rowSum.GetGenericIndexer<int>();
            while (top < H && rs[top, 0] < minInkRow) top++;
            while (bottom > top && rs[bottom, 0] < minInkRow) bottom--;

            int bw = Math.Max(1, right - left + 1);
            int bh = Math.Max(1, bottom - top + 1);
            int padX = (int)Math.Round(bw * padXRatio);
            int padY = (int)Math.Round(bh * padYRatio);

            int x = Math.Max(0, left - padX);
            int y = Math.Max(0, top - padY);
            int w = Math.Min(W - x, bw + 2 * padX);
            int h = Math.Min(H - y, bh + 2 * padY);

            return new Mat(src, new Rect(x, y, w, h)).Clone();
        }

        /// <summary>
        /// Runs a single instance of the CTC (Connectionist Temporal Classification) recognizer.
        /// </summary>
        /// <param name="rec">The inference session used to run the recognition model.</param>
        /// <param name="vocab">The vocabulary string used for decoding the logits into text.</param>
        /// <param name="blank">The index representing the blank token in the vocabulary.</param>
        /// <param name="input">The input tensor containing the features to be recognized.</param>
        /// <returns>A string representing the decoded text from the input features.</returns>
        private string RunRecognizerCtcSingle(InferenceSession rec, string vocab, int blank, DenseTensor<float> input)
        {
            var inpName = rec.InputMetadata.Keys.First();
            using var results = rec.Run(new[] { NamedOnnxValue.CreateFromTensor(inpName, input) });
            var logits = results.First().AsTensor<float>();
            return CtcGreedyDecode(logits, vocab, blank);
        }

        /// <summary>
        /// Recognizes text for each rotated rectangle (box) on the provided page image.
        /// The method returns a list with one element per input box:
        /// - null for sentinel boxes (center X/Y < 0),
        /// - empty string if the crop was empty,
        /// - otherwise the recognized text for that box.
        /// </summary>
        /// <param name="pageBgr">Source page image in BGR color space (OpenCvSharp <see cref="Mat"/>).</param>
        /// <param name="boxes">List of rotated rectangles specifying regions to recognize.</param>
        /// <param name="saved">Unused debug counter (reserved for debug saving of crops).</param>
        /// <param name="targetH">Target height (pixels) used when resizing word crops for the recognizer.</param>
        /// <returns>
        /// A list of recognized strings (one element per input box). Elements may be null, empty, or a non-empty string.
        /// </returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="pageBgr"/> or <paramref name="boxes"/> is null.</exception>
        /// <exception cref="ArgumentException">Thrown when <paramref name="pageBgr"/> is empty.</exception>
        /// <exception cref="InvalidOperationException">Thrown when internal preprocessing or model inference fails; the inner exception is preserved.</exception>
        internal List<string?> RecognizePerBox(Mat pageBgr, List<RotatedRect> boxes, int saved = 0, int targetH = 32)
        {
            if (pageBgr is null) throw new ArgumentNullException(nameof(pageBgr));
            if (boxes is null) throw new ArgumentNullException(nameof(boxes));
            if (pageBgr.Empty()) throw new ArgumentException("Input image is empty.", nameof(pageBgr));

            var results = new List<string?>(boxes.Count);
            if (boxes.Count == 0)
                return results;

            try
            {
                for (int i = 0; i < boxes.Count; i++)
                {
                    var rr = boxes[i];

                    if (rr.Center.X < 0 && rr.Center.Y < 0)
                    {
                        results.Add(null);
                        continue;
                    }

                    using var crop0 = CropAabb(pageBgr, rr, padXRatio: 0.05f, padYRatio: 0.30f);
                    if (crop0.Empty())
                    {
                        results.Add(string.Empty);
                        continue;
                    }

                    using var crop = TrimWhitespace(crop0, padXRatio: 0.15f, padYRatio: 0.25f);
                    var ten = PreprocessWordDynamic(crop, targetH: targetH, maxW: 512);
                    var text = RunRecognizerCtcSingle(_session, _vocab, _blankIndex, ten);
                    results.Add(text);
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException("RecognizePerBox failed during preprocessing or inference.", ex);
            }

            return results;
        }

        public void Dispose() => _session.Dispose();
    }
}
