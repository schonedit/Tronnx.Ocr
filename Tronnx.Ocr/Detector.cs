using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using Tronnx.Ocr.Types;

namespace Tronnx.Ocr
{
    internal class Detector : IDisposable
    {
        readonly InferenceSession _session;

        internal Detector(string modelPath)
        {
            var so = new SessionOptions();
            so.AppendExecutionProvider_CPU();
            so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            _session = new InferenceSession(modelPath, so);
        }

        /// <summary>
        /// Preprocesses an image for a detector by resizing and padding it to a target size while maintaining the aspect
        /// ratio.
        /// </summary>
        /// <param name="bgr">The input image in BGR format.</param>
        /// <param name="target">The target size for the output image, which will be a square of dimensions target x target.</param>
        /// <param name="scale">The scaling factor applied to the input image to fit within the target size.</param>
        /// <param name="padX">The horizontal padding added to center the image within the target size.</param>
        /// <param name="padY">The vertical padding added to center the image within the target size.</param>
        /// <returns>A <see cref="DenseTensor{T}"/> representing the preprocessed image in RGB format, normalized to a scale of 0 to
        /// 1. The tensor has dimensions [1, 3, target, target], where 1 is the batch size, 3 is the number of color
        /// channels, and target x target is the size of the image.</returns>
        internal DenseTensor<float> PreprocessForDetector(Mat bgr, int target, out float scale, out int padX, out int padY)
        {
            int h = bgr.Rows, w = bgr.Cols;
            float s = Math.Min((float)target / w, (float)target / h);
            int nw = (int)Math.Round(w * s);
            int nh = (int)Math.Round(h * s);
            padX = (target - nw) / 2;
            padY = (target - nh) / 2;
            scale = s;

            using var resized = new Mat();
            Cv2.Resize(bgr, resized, new Size(nw, nh), 0, 0, InterpolationFlags.Linear);
            using var rgb = new Mat();
            Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);

            using var canvas = new Mat(target, target, MatType.CV_8UC3, Scalar.Black);
            rgb.CopyTo(new Mat(canvas, new Rect(padX, padY, nw, nh)));

            var tensor = new DenseTensor<float>(new[] { 1, 3, target, target });
            var idx = canvas.GetGenericIndexer<Vec3b>();
            for (int y = 0; y < target; y++)
            {
                for (int x = 0; x < target; x++)
                {
                    var p = idx[y, x];
                    tensor[0, 0, y, x] = p.Item0 / 255f; // R
                    tensor[0, 1, y, x] = p.Item1 / 255f; // G
                    tensor[0, 2, y, x] = p.Item2 / 255f; // B
                }
            }
            return tensor;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        internal Tensor<float> RunDetector(DenseTensor<float> input)
        {
            var inpName = _session.InputMetadata.Keys.First();
            using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inpName, input) });
            return results.First().AsTensor<float>();
        }

        /// <summary>
        /// Processes the output logits from a detection model to extract rotated bounding boxes.
        /// </summary>
        /// <remarks>The method processes the logits to generate a probability map, applies a threshold to filter
        /// out low-probability detections, and then extracts contours to determine the bounding boxes. Only boxes with an
        /// area greater than <paramref name="minBoxArea"/> are included in the result.</remarks>
        /// <param name="logits">The tensor containing the detection model's output logits. Expected dimensions are either [1, C, H, W] or [1, H,
        /// W].</param>
        /// <param name="probThresh">The probability threshold for filtering detections. Values range from 0 to 1. Default is 0.3.</param>
        /// <param name="minBoxArea">The minimum area required for a detected box to be considered valid. Default is 30.</param>
        /// <returns>A list of <see cref="RotatedRect"/> representing the detected bounding boxes that meet the specified criteria.</returns>
        /// <exception cref="InvalidOperationException">Thrown if the dimensions of <paramref name="logits"/> are not as expected.</exception>
        internal List<RotatedRect> PostprocessBoxes(Tensor<float> logits, double probThresh = 0.3, double minBoxArea = 30)
        {
            try
            {
                float minVal = logits.Min();
                float maxVal = logits.Max();
                float meanVal = logits.Average();
                Console.WriteLine($"[detector raw logits] min={minVal:F3}, max={maxVal:F3}, mean={meanVal:F3}");
            }
            catch (Exception ex)
            {
                Console.WriteLine("logits range error: " + ex.Message);
            }

            int H, W; float[] map;
            if (logits.Dimensions.Length == 4)
            {
                int C = logits.Dimensions[1];
                H = logits.Dimensions[2];
                W = logits.Dimensions[3];
                map = new float[H * W];
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++)
                        map[y * W + x] = logits[0, 0, y, x];
            }
            else if (logits.Dimensions.Length == 3)
            {
                H = logits.Dimensions[1];
                W = logits.Dimensions[2];
                map = new float[H * W];
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++)
                        map[y * W + x] = logits[0, y, x];
            }
            else throw new InvalidOperationException($"Váratlan detektor kimenet: [{string.Join(",", logits.Dimensions.ToArray())}]");

            float min = map.Min(), max = map.Max();
            if (min < 0f || max > 1f)
            {
                for (int i = 0; i < map.Length; i++)
                    map[i] = MathF.Pow(map[i], 1.3f);
            }

            using var prob = new Mat(H, W, MatType.CV_32FC1);
            var pidx = prob.GetGenericIndexer<float>();
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    pidx[y, x] = map[y * W + x];

            using var prob8 = new Mat();
            prob.ConvertTo(prob8, MatType.CV_8UC1, 255.0);
            Cv2.Threshold(prob8, prob8, probThresh * 255.0, 255, ThresholdTypes.Binary);
            using var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3));
            Cv2.MorphologyEx(prob8, prob8, MorphTypes.Dilate, kernel, iterations: 2);

            //using (var normFloat = new Mat())
            //{
            //    Cv2.Normalize(prob, normFloat, 0, 1, NormTypes.MinMax);
            //    Cv2.ImWrite(Path.Combine(DEBUG_DIR, "heatmap_gray.png"), normFloat * 255);
            //}

            //using (var sw = new StreamWriter(Path.Combine(DEBUG_DIR, "heatmap_values.csv")))
            //{
            //    int stepY = Math.Max(1, prob.Rows / 100);
            //    int stepX = Math.Max(1, prob.Cols / 100);
            //    for (int y = 0; y < prob.Rows; y += stepY)
            //    {
            //        for (int x = 0; x < prob.Cols; x += stepX)
            //        {
            //            float val = pidx[y, x];
            //            sw.Write(val.ToString("F3") + ";");
            //        }
            //        sw.WriteLine();
            //    }
            //}
            //Cv2.ImWrite("detector_mask_debug.png", prob8);

            Cv2.FindContours(prob8, out Point[][] contours, out HierarchyIndex[] hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            var boxes = new List<RotatedRect>(contours.Length);
            foreach (var cont in contours)
            {
                if (cont.Length < 3) continue;
                var rr = Cv2.MinAreaRect(cont);
                if (rr.Size.Width * rr.Size.Height < minBoxArea) continue;
                boxes.Add(rr);
            }
            return boxes;
        }

        /// <summary>
        /// Remaps a list of rotated rectangles to their original coordinates by reversing scaling and padding
        /// transformations.
        /// </summary>
        /// <param name="boxes">The list of <see cref="RotatedRect"/> objects to be remapped.</param>
        /// <param name="scale">The scale factor that was applied to the original coordinates. Must be a positive number.</param>
        /// <param name="padX">The horizontal padding that was added to the original coordinates.</param>
        /// <param name="padY">The vertical padding that was added to the original coordinates.</param>
        /// <returns>A list of <see cref="RotatedRect"/> objects with coordinates remapped to the original scale and position.</returns>
        internal List<RotatedRect> RemapBoxesToOriginal(List<RotatedRect> boxes, float scale, int padX, int padY)
        {
            var mapped = new List<RotatedRect>(boxes.Count);
            foreach (var rr in boxes)
            {
                var pts = rr.Points();
                for (int i = 0; i < pts.Length; i++)
                {
                    pts[i].X = (pts[i].X - padX) / scale;
                    pts[i].Y = (pts[i].Y - padY) / scale;
                }
                var rr2 = Cv2.MinAreaRect(pts);
                mapped.Add(rr2);
            }
            return mapped;
        }

        /// <summary>
        /// Orders a list of rotated rectangles to reflect a reading order, typically from top-left to bottom-right.
        /// </summary>
        /// <remarks>The method modifies the input list to arrange the rectangles in a reading order, which is
        /// determined by the top-left corner of each rectangle. Rectangles are grouped into rows based on their vertical
        /// position, and each row is ordered from left to right. The method assumes that the rectangles are aligned in a
        /// way that allows for such ordering.</remarks>
        /// <param name="boxes">A list of <see cref="RotatedRect"/> objects representing the rectangles to be ordered.</param>
        internal void OrderReading(List<RotatedRect> boxes)
        {
            var boxPoints = new List<BoxPoint>();
            foreach (var rr in boxes)
            {
                var pts = rr.Points().Select(p => new Point((int)Math.Round(p.X), (int)Math.Round(p.Y))).ToArray();
                var topLeft = pts.OrderBy(p => p.Y).ThenBy(p => p.X).First();
                var topRight = pts.OrderBy(p => p.X).ThenByDescending(p => p.Y).First();
                var bottomLeft = pts.OrderByDescending(p => p.X).ThenBy(p => p.Y).First();
                var bottomRight = pts.OrderByDescending(p => p.Y).ThenByDescending(p => p.X).First();

                // topleft.....xx......topRight
                // |                     |
                // Y                     Y
                // |                     |
                // bottomLeft....xx....bottomRight

                var boxPoint = new BoxPoint()
                {
                    TopLeft = topLeft,
                    TopRight = topRight,
                    BottomRight = bottomRight,
                    BottomLeft = bottomLeft,
                    Box = rr
                };
                boxPoints.Add(boxPoint);
            }

            if (boxPoints.Count == 0) return;

            var firstBox = boxPoints.Where(x => x.TopLeft.Y == boxPoints.Min(b => b.TopLeft.Y)).OrderBy(x => x.TopLeft.X).First();
            var boxRows = new List<List<BoxPoint>>();
            var orderedBoxes = new List<RotatedRect>();
            bool first = true;
            boxPoints = boxPoints.OrderBy(x => x.TopLeft.Y).ThenBy(x => x.TopLeft.X).ToList();

            double treshold = CalculateLineThreshold(boxes, ratio: 0.5);
            foreach (var box in boxPoints)
            {
                if (first)
                {
                    boxRows.Add(new List<BoxPoint>() { firstBox });
                    first = false;
                }
                else
                {
                    if (Math.Abs(boxRows.Last().Last().TopLeft.Y - box.TopLeft.Y) <= treshold)
                    {
                        boxRows.Last().Add(box);
                    }
                    else
                    {
                        boxRows.Add(new List<BoxPoint>() { box });
                    }
                }
            }

            foreach (var item in boxRows)
            {
                var list = item.OrderBy(x => x.TopLeft.X).ToList();
                foreach (var box in list)
                {
                    orderedBoxes.Add(box.Box);
                }
                orderedBoxes.Add(new RotatedRect(new Point2f(-1, -1), new Size2f(0, 0), 0));
            }
            boxes.Clear();
            boxes.AddRange(orderedBoxes);
        }

        /// <summary>
        /// Calculates the line threshold based on the median height of a collection of rotated rectangles.
        /// </summary>
        /// <param name="boxes">A list of <see cref="RotatedRect"/> objects representing the rectangles to analyze. Must contain at least three
        /// elements.</param>
        /// <param name="ratio">A multiplier applied to the median height to determine the threshold. Defaults to 0.5.</param>
        /// <returns>The calculated line threshold as a double. Returns 0 if the list contains fewer than three rectangles.</returns>
        private double CalculateLineThreshold(List<RotatedRect> boxes, double ratio = 0.5)
        {
            if (boxes == null || boxes.Count < 3)
                return 0;

            var heights = boxes.Select(b => Math.Min(b.Size.Height, b.Size.Width)).OrderBy(h => h).ToList();
            double medianHeight;
            int count = heights.Count;
            if (count % 2 == 0)
            {
                medianHeight = (heights[count / 2 - 1] + heights[count / 2]) / 2.0;
            }
            else
            {
                medianHeight = heights[count / 2];
            }
            return medianHeight * ratio;
        }


        /// <summary>
        /// Saves an image with overlayed rectangles to the specified file path.
        /// </summary>
        /// <remarks>Each rectangle in <paramref name="boxes"/> is drawn with a green outline on a clone of the
        /// <paramref name="pageBgr"/> image. The method writes the modified image to the specified <paramref
        /// name="outPath"/>.</remarks>
        /// <param name="pageBgr">The source image on which the rectangles will be drawn. This image is not modified.</param>
        /// <param name="boxes">A list of <see cref="RotatedRect"/> objects representing the rectangles to be drawn on the image.</param>
        /// <param name="outPath">The file path where the resulting image with overlays will be saved. Must be a valid path.</param>
        private void SaveOverlay(Mat pageBgr, List<RotatedRect> boxes, string outPath)
        {
            using var canvas = pageBgr.Clone();
            foreach (var rr in boxes)
            {
                var pts = rr.Points().Select(p => new Point((int)Math.Round(p.X), (int)Math.Round(p.Y))).ToArray();
                Cv2.Polylines(canvas, new[] { pts }, true, new Scalar(0, 255, 0), 2);
                Console.WriteLine($"Center=({rr.Center.X:F1}, {rr.Center.Y:F1}) " +
                      $"Size=({rr.Size.Width:F1}x{rr.Size.Height:F1})");
            }
            Cv2.ImWrite(outPath, canvas);
        }

        /// <summary>
        /// Saves an heatmap image to the specified file path.
        /// </summary>
        /// <param name="detOutput"></param>
        private void GenerateHeatMap(Tensor<float> detOutput)
        {
            try
            {
                int H, W;
                Mat prob;

                if (detOutput.Dimensions.Length == 4)
                {
                    H = detOutput.Dimensions[2];
                    W = detOutput.Dimensions[3];
                    prob = new Mat(H, W, MatType.CV_32FC1);
                    var idx = prob.GetGenericIndexer<float>();
                    for (int y = 0; y < H; y++)
                        for (int x = 0; x < W; x++)
                            idx[y, x] = detOutput[0, 0, y, x];
                }
                else if (detOutput.Dimensions.Length == 3)
                {
                    H = detOutput.Dimensions[1];
                    W = detOutput.Dimensions[2];
                    prob = new Mat(H, W, MatType.CV_32FC1);
                    var idx = prob.GetGenericIndexer<float>();
                    for (int y = 0; y < H; y++)
                        for (int x = 0; x < W; x++)
                            idx[y, x] = detOutput[0, y, x];
                }
                else
                {
                    goto EndHeat;
                }

                using (prob)
                using (var norm = new Mat())
                using (var heat8 = new Mat())
                {
                    Cv2.Normalize(prob, norm, 0, 255, NormTypes.MinMax);
                    norm.ConvertTo(heat8, MatType.CV_8UC1);
                    Cv2.ApplyColorMap(heat8, heat8, ColormapTypes.Jet);
                    string DEBUG_DIR = "debug_out";
                    var outPath = Path.Combine(DEBUG_DIR, "heatmap.jpg");
                    Cv2.ImWrite(outPath, heat8);
                    Console.WriteLine("HeatMap saved: " + outPath);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("HeatMap export failed: " + ex.Message);
            }
        EndHeat:;
        }
        public void Dispose() => _session.Dispose();
    }
}
