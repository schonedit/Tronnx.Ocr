namespace Tronnx.Ocr;

/// <summary>
/// Olvasási sorrend: fentről lefelé, egy fizikai soron belül balról jobbra.
/// Két doboz akkor van egy sorban, ha a függőleges középpontjuk távolsága kisebb,
/// mint a kisebbik magasság fele.
/// </summary>
internal static class ReadingOrder
{
    public static bool SameRow(Quad a, Quad b)
    {
        float tolerance = 0.5f * MathF.Min(a.Height, b.Height);
        return MathF.Abs(a.Center.Y - b.Center.Y) < tolerance;
    }

    public static List<OcrLine> Sort(IEnumerable<OcrLine> lines)
    {
        var byY = lines.OrderBy(l => l.Box.Center.Y).ToList();
        var result = new List<OcrLine>(byY.Count);
        int i = 0;
        while (i < byY.Count)
        {
            // Egy fizikai sor összegyűjtése az első elemhez képest.
            var row = new List<OcrLine> { byY[i] };
            int j = i + 1;
            while (j < byY.Count && SameRow(byY[i].Box, byY[j].Box))
                row.Add(byY[j++]);
            result.AddRange(row.OrderBy(l => l.Box.Center.X));
            i = j;
        }
        return result;
    }
}
