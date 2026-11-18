namespace Tronnx.Ocr.Detection;

/// <summary>Egy detektált szövegdoboz a detektor bemenetének koordinátáiban (folytonos tér).</summary>
internal readonly record struct RotatedBox(
    float CenterX, float CenterY,
    float AxisX, float AxisY,       // a szöveg irányú egységvektor (u)
    float HalfWidth, float HalfHeight,
    float Score);

/// <summary>
/// A DB (Differentiable Binarization) detektor valószínűségi térképéből szövegdobozokat készít.
/// A PaddleOCR Python-kódja OpenCV-t (findContours, minAreaRect) és pyclipper-t használ;
/// itt ugyanezt tiszta C#-ban csináljuk, natív függőség nélkül:
///   1. küszöbölés → bináris maszk
///   2. összefüggő komponensek (8-szomszédság), komponensenként soronkénti min/max x
///   3. konvex burok → minimális területű forgatott téglalap
///   4. pontszám (átlagos valószínűség), szűrés
///   5. „unclip": a téglalap kitágítása (a DB a szövegmagot jósolja, nem a teljes betűt)
/// </summary>
internal sealed class DbPostProcessor
{
    private readonly float _threshold;
    private readonly float _boxThreshold;
    private readonly float _unclipRatio;
    private readonly int _minSize;
    private readonly int _maxCandidates;

    public DbPostProcessor(float threshold, float boxThreshold, float unclipRatio, int minSize = 3, int maxCandidates = 1000)
    {
        _threshold = threshold;
        _boxThreshold = boxThreshold;
        _unclipRatio = unclipRatio;
        _minSize = minSize;
        _maxCandidates = maxCandidates;
    }

    public List<RotatedBox> Process(ReadOnlySpan<float> prob, int width, int height)
    {
        var boxes = new List<RotatedBox>();
        int size = width * height;

        // 0 = háttér, 1 = szöveg (még nem bejárt), 2 = már bejárt
        var state = new byte[size];
        for (int i = 0; i < size; i++)
            state[i] = prob[i] > _threshold ? (byte)1 : (byte)0;

        var stack = new int[1024];
        var rowMin = new int[height];
        var rowMax = new int[height];
        Array.Fill(rowMin, int.MaxValue);
        Array.Fill(rowMax, -1);
        var touchedRows = new List<int>();
        var hullInput = new List<(float X, float Y)>();

        for (int start = 0; start < size && boxes.Count < _maxCandidates; start++)
        {
            if (state[start] != 1)
                continue;

            // --- 2. elárasztásos kitöltés (iteratív, saját veremmel – nincs rekurzió, nincs List<Point>) ---
            int count = 0;
            double probSum = 0;
            int sp = 0;
            stack[sp++] = start;
            state[start] = 2;

            while (sp > 0)
            {
                int idx = stack[--sp];
                int cy = idx / width, cx = idx - cy * width;
                count++;
                probSum += prob[idx];

                if (rowMax[cy] < 0)
                    touchedRows.Add(cy);
                if (cx < rowMin[cy]) rowMin[cy] = cx;
                if (cx > rowMax[cy]) rowMax[cy] = cx;

                for (int dy = -1; dy <= 1; dy++)
                {
                    int ny = cy + dy;
                    if ((uint)ny >= (uint)height) continue;
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int nx = cx + dx;
                        if ((uint)nx >= (uint)width) continue;
                        int n = ny * width + nx;
                        if (state[n] != 1) continue;
                        state[n] = 2;
                        if (sp == stack.Length)
                            Array.Resize(ref stack, stack.Length * 2);
                        stack[sp++] = n;
                    }
                }
            }

            // A konvex burokhoz elég soronként a két szélső pixel.
            hullInput.Clear();
            foreach (int y in touchedRows)
            {
                hullInput.Add((rowMin[y], y));
                if (rowMax[y] != rowMin[y])
                    hullInput.Add((rowMax[y], y));
                rowMin[y] = int.MaxValue;
                rowMax[y] = -1;
            }
            touchedRows.Clear();

            if (count < 4)
                continue;

            float score = (float)(probSum / count);
            if (score < _boxThreshold)
                continue;

            var hull = ConvexHull(hullInput);
            if (hull.Count < 3)
                continue;

            var box = MinAreaRect(hull);
            // Pixelközép → pixelterület: mindkét irányban fél pixel hozzáadása.
            float hw = box.HalfWidth + 0.5f, hh = box.HalfHeight + 0.5f;
            if (MathF.Min(hw, hh) * 2 < _minSize)
                continue;

            // --- 5. unclip: d = terület · arány / kerület (a pyclipper-es eltolás téglalapra) ---
            float w2 = hw * 2, h2 = hh * 2;
            float d = w2 * h2 * _unclipRatio / (2 * (w2 + h2));
            hw += d;
            hh += d;
            if (MathF.Min(hw, hh) * 2 < _minSize + 2)
                continue;

            // Pixelindex-tér → folytonos tér (pixelközép = i + 0,5)
            boxes.Add(box with { CenterX = box.CenterX + 0.5f, CenterY = box.CenterY + 0.5f, HalfWidth = hw, HalfHeight = hh, Score = score });
        }

        return boxes;
    }

    /// <summary>Monotone chain (Andrew-algoritmus), O(n log n).</summary>
    internal static List<(float X, float Y)> ConvexHull(List<(float X, float Y)> points)
    {
        var pts = points.Distinct().OrderBy(p => p.X).ThenBy(p => p.Y).ToList();
        if (pts.Count < 3)
            return pts;

        static float Cross((float X, float Y) o, (float X, float Y) a, (float X, float Y) b)
            => (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);

        var hull = new List<(float X, float Y)>(pts.Count * 2);
        foreach (var p in pts)
        {
            while (hull.Count >= 2 && Cross(hull[^2], hull[^1], p) <= 0) hull.RemoveAt(hull.Count - 1);
            hull.Add(p);
        }
        int lower = hull.Count + 1;
        for (int i = pts.Count - 2; i >= 0; i--)
        {
            var p = pts[i];
            while (hull.Count >= lower && Cross(hull[^2], hull[^1], p) <= 0) hull.RemoveAt(hull.Count - 1);
            hull.Add(p);
        }
        hull.RemoveAt(hull.Count - 1);
        return hull;
    }

    /// <summary>
    /// Minimális területű befoglaló téglalap: a téglalap egyik oldala mindig a konvex burok
    /// valamelyik élére esik, ezért elég az éleket végigpróbálni.
    /// Az eredmény tengelyét a szöveg irányához igazítjuk: alapból a vízszinteshez közelebbi
    /// tengely a szövegirány; csak határozottan álló doboznál (≥ 1,5× magasabb) fordítunk.
    /// </summary>
    internal static RotatedBox MinAreaRect(List<(float X, float Y)> hull)
    {
        float bestArea = float.MaxValue;
        float bux = 1, buy = 0, bu0 = 0, bu1 = 0, bv0 = 0, bv1 = 0;

        for (int i = 0; i < hull.Count; i++)
        {
            var p = hull[i];
            var q = hull[(i + 1) % hull.Count];
            float ex = q.X - p.X, ey = q.Y - p.Y;
            float len = MathF.Sqrt(ex * ex + ey * ey);
            if (len < 1e-6f) continue;
            float ux = ex / len, uy = ey / len;
            float vx = -uy, vy = ux;

            float u0 = float.MaxValue, u1 = float.MinValue, v0 = float.MaxValue, v1 = float.MinValue;
            foreach (var h in hull)
            {
                float u = h.X * ux + h.Y * uy;
                float v = h.X * vx + h.Y * vy;
                if (u < u0) u0 = u;
                if (u > u1) u1 = u;
                if (v < v0) v0 = v;
                if (v > v1) v1 = v;
            }
            float area = (u1 - u0) * (v1 - v0);
            if (area < bestArea)
            {
                bestArea = area;
                bux = ux; buy = uy; bu0 = u0; bu1 = u1; bv0 = v0; bv1 = v1;
            }
        }

        float bvx = -buy, bvy = bux;
        float cu = (bu0 + bu1) / 2, cv = (bv0 + bv1) / 2;
        float cx = cu * bux + cv * bvx, cy = cu * buy + cv * bvy;
        float hw = (bu1 - bu0) / 2, hh = (bv1 - bv0) / 2;

        // Tengely kiválasztása: u legyen a vízszinteshez közelebbi.
        if (MathF.Abs(buy) > MathF.Abs(bux))
        {
            (bux, buy) = (bvx, bvy);
            (hw, hh) = (hh, hw);
        }
        // Határozottan álló doboz (függőleges szöveg) → a hosszabb tengely a szövegirány.
        if (hh >= 1.5f * hw)
        {
            (bux, buy) = (-buy, bux);
            (hw, hh) = (hh, hw);
        }
        // Irány: balról jobbra (függőlegesnél fentről lefelé).
        if (bux < 0 || (MathF.Abs(bux) < 1e-6f && buy < 0))
        {
            bux = -bux;
            buy = -buy;
        }

        return new RotatedBox(cx, cy, bux, buy, hw, hh, 0f);
    }
}
