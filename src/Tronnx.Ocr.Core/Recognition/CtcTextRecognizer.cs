using System.Buffers;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Tronnx.Ocr.Imaging;

namespace Tronnx.Ocr.Recognition;

/// <summary>
/// Szövegsor-felismerés CTC-dekódolással (PP-OCRv5 rec). Bemenet: [N, 3, 48, W],
/// kimenet: [N, T, C] valószínűségek, ahol T ≈ W/8 időlépés, C = 1 (blank) + szótár (+ szóköz).
/// </summary>
internal sealed class CtcTextRecognizer : IDisposable
{
    private const int MinBatchWidth = 320;     // a PaddleOCR is legalább 320 széles bemenettel fut
    private const int MaxLineWidth = 3200;     // ennél hosszabb sort vízszintesen összenyomunk

    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string _outputName;
    private readonly string[] _labels;         // 0 = CTC blank
    private readonly int _height;
    private readonly int _batchSize;
    private readonly int _parallelism;

    public CtcTextRecognizer(InferenceSession session, IReadOnlyList<string> dictionary, OcrEngineOptions options, int parallelism)
    {
        _session = session;
        _inputName = session.InputMetadata.Keys.First();
        _outputName = session.OutputMetadata.Keys.First();
        _batchSize = Math.Max(1, options.RecognitionBatchSize);
        _parallelism = Math.Max(1, parallelism);

        var inDims = session.InputMetadata[_inputName].Dimensions; // [-1, 3, 48, -1]
        _height = inDims.Length == 4 && inDims[2] > 0 ? inDims[2] : 48;

        // Az osztályok száma a modellből: ebből derül ki, hogy a szóköz külön osztály-e.
        var outDims = session.OutputMetadata[_outputName].Dimensions;
        int classes = outDims[^1];
        var labels = new List<string>(dictionary.Count + 2) { "" };
        labels.AddRange(dictionary);
        if (classes == dictionary.Count + 2)
            labels.Add(" ");
        else if (classes > 0 && classes != dictionary.Count + 1)
            throw new InvalidOperationException(
                $"Recognition model has {classes} output classes, but the dictionary has {dictionary.Count} entries " +
                $"(expected {dictionary.Count + 1} or {dictionary.Count + 2}). Wrong dictionary for this model?");
        _labels = labels.ToArray();
    }

    public OcrLine[] Recognize(BgrImage image, IReadOnlyList<Quad> boxes)
    {
        var results = new OcrLine[boxes.Count];
        if (boxes.Count == 0)
            return results;

        // Célszélesség soronként (arányt tartva 48 px magasra). Rendezés szélesség szerint,
        // hogy egy kötegbe hasonló hosszú sorok kerüljenek → kevesebb üres kitöltés.
        var widths = new int[boxes.Count];
        for (int i = 0; i < boxes.Count; i++)
        {
            float h = MathF.Max(1f, boxes[i].Height);
            widths[i] = Math.Clamp((int)MathF.Ceiling(_height * boxes[i].Width / h), 8, MaxLineWidth);
        }
        var order = Enumerable.Range(0, boxes.Count).OrderBy(i => widths[i]).ToArray();

        if (_parallelism == 1)
        {
            for (int start = 0; start < order.Length; start += _batchSize)
            {
                int n = Math.Min(_batchSize, order.Length - start);
                RecognizeBatch(image, boxes, widths, order.AsSpan(start, n), results);
            }
            return results;
        }

        // Párhuzamos mód: a munkamenet 1 szálon fut, a párhuzamosság a sorok között van.
        // Mérés (6 mag): egy hívásban az ORT 1 → 6 szálon csak 1,3×-ot gyorsult, tehát a magok
        // nagy része állt. Itt minden mag külön köteget dolgoz fel.
        //
        // Kötegméret: elég kicsi, hogy minden magnak jusson munka (és a végén ne egy hosszú köteg
        // várasson), de ≤ RecognitionBatchSize. Kb. két köteg jut egy magra a terheléselosztáshoz.
        int size = Math.Clamp((order.Length + 2 * _parallelism - 1) / (2 * _parallelism), 1, _batchSize);
        var starts = new List<int>();
        for (int start = 0; start < order.Length; start += size)
            starts.Add(start);
        starts.Reverse(); // a leghosszabb sorok kötegei indulnak elsőként → kiegyenlítettebb befejezés

        Parallel.ForEach(
            starts,
            new ParallelOptions { MaxDegreeOfParallelism = _parallelism },
            start =>
            {
                int n = Math.Min(size, order.Length - start);
                RecognizeBatch(image, boxes, widths, order.AsSpan(start, n), results);
            });
        return results;
    }

    private void RecognizeBatch(BgrImage image, IReadOnlyList<Quad> boxes, int[] widths, ReadOnlySpan<int> batch, OcrLine[] results)
    {
        int batchWidth = MinBatchWidth;
        foreach (int i in batch)
            batchWidth = Math.Max(batchWidth, widths[i]);

        int plane = _height * batchWidth;
        int itemSize = 3 * plane;
        int total = itemSize * batch.Length;
        float[] buffer = ArrayPool<float>.Shared.Rent(total);
        try
        {
            var tensor = buffer.AsSpan(0, total);
            tensor.Clear(); // a jobb oldali kitöltés 0 (normalizált térben)

            for (int k = 0; k < batch.Length; k++)
            {
                int i = batch[k];
                var q = boxes[i];
                var u = q.TopRight - q.TopLeft;     // szövegirány
                var v = q.BottomLeft - q.TopLeft;   // lefelé
                int w = widths[i];
                Sampler.SampleToChw(image, q.TopLeft, u * (1f / w), v * (1f / _height),
                    w, _height, tensor.Slice(k * itemSize, itemSize), batchWidth, plane, ChannelNorm.MinusOneToOne);
            }

            using var input = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance,
                buffer.AsMemory(0, total), new long[] { batch.Length, 3, _height, batchWidth });
            using var runOptions = new RunOptions();
            using var outputs = _session.Run(runOptions, new[] { _inputName }, new[] { input }, new[] { _outputName });

            var probs = outputs[0].GetTensorDataAsSpan<float>();
            var shape = outputs[0].GetTensorTypeAndShape().Shape; // [N, T, C]
            int steps = (int)shape[1], classes = (int)shape[2];
            float stride = (float)batchWidth / steps; // egy időlépés hány bemeneti pixel

            for (int k = 0; k < batch.Length; k++)
            {
                int i = batch[k];
                results[i] = Decode(probs.Slice(k * steps * classes, steps * classes), steps, classes, stride, widths[i], boxes[i]);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// Mohó (greedy) CTC-dekódolás: időlépésenként a legvalószínűbb osztály; az egymás utáni
    /// ismétlések és a blank kiesnek. Közben megjegyezzük, melyik időlépésnél jelent meg a karakter –
    /// ebből becsüljük a szavak helyét a soron belül (kiemeléshez és kereshető PDF-hez).
    /// </summary>
    private OcrLine Decode(ReadOnlySpan<float> probs, int steps, int classes, float stride, int itemWidth, Quad box)
    {
        var text = new StringBuilder();
        var words = new List<OcrWord>();
        var word = new StringBuilder();
        int wordStart = -1, wordEnd = -1;
        float wordConf = 0, lineConf = 0;
        int wordChars = 0, lineChars = 0, last = 0;

        void FlushWord()
        {
            if (word.Length == 0) return;
            // Időlépés → arány a sor szélességén belül; fél lépés ráhagyás mindkét oldalon.
            float f0 = (wordStart - 0.5f) * stride / itemWidth;
            float f1 = (wordEnd + 1.5f) * stride / itemWidth;
            words.Add(new OcrWord(word.ToString(), wordConf / wordChars, box.SliceAlongWidth(f0, f1)));
            word.Clear();
            wordConf = 0;
            wordChars = 0;
            wordStart = -1;
        }

        for (int t = 0; t < steps; t++)
        {
            var row = probs.Slice(t * classes, classes);
            int best = 0;
            float bestP = row[0];
            for (int c = 1; c < row.Length; c++)
            {
                if (row[c] > bestP) { bestP = row[c]; best = c; }
            }

            if (best != 0 && best != last && best < _labels.Length)
            {
                string ch = _labels[best];
                text.Append(ch);
                lineConf += bestP;
                lineChars++;

                if (ch == " ")
                {
                    FlushWord();
                }
                else
                {
                    if (wordStart < 0) wordStart = t;
                    wordEnd = t;
                    word.Append(ch);
                    wordConf += bestP;
                    wordChars++;
                }
            }
            last = best;
        }
        FlushWord();

        string lineText = text.ToString().Trim();
        return new OcrLine(lineText, lineChars > 0 ? lineConf / lineChars : 0f, box, words);
    }

    public void Dispose() => _session.Dispose();
}
