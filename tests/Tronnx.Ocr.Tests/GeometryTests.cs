using Tronnx.Ocr.Detection;
using Xunit;

namespace Tronnx.Ocr.Tests;

public class GeometryTests
{
    [Fact]
    public void ConvexHull_DropsInteriorPoints()
    {
        var pts = new List<(float X, float Y)> { (0, 0), (10, 0), (10, 5), (0, 5), (5, 2), (3, 3) };
        var hull = DbPostProcessor.ConvexHull(pts);
        Assert.Equal(4, hull.Count);
        Assert.DoesNotContain((5f, 2f), hull);
    }

    [Fact]
    public void MinAreaRect_HorizontalLine_KeepsTextDirection()
    {
        // 100 × 10-es vízszintes sáv
        var hull = DbPostProcessor.ConvexHull(new() { (0, 0), (100, 0), (100, 10), (0, 10) });
        var r = DbPostProcessor.MinAreaRect(hull);
        Assert.Equal(1f, r.AxisX, 3);
        Assert.Equal(0f, r.AxisY, 3);
        Assert.Equal(50f, r.HalfWidth, 3);
        Assert.Equal(5f, r.HalfHeight, 3);
        Assert.Equal(50f, r.CenterX, 3);
        Assert.Equal(5f, r.CenterY, 3);
    }

    [Fact]
    public void MinAreaRect_SquareishBox_StaysHorizontal()
    {
        // Egy magas, de nem „nagyon magas" doboz (pl. egyetlen „7"-es): nem szabad elforgatni.
        var hull = DbPostProcessor.ConvexHull(new() { (0, 0), (10, 0), (10, 14), (0, 14) });
        var r = DbPostProcessor.MinAreaRect(hull);
        Assert.Equal(1f, r.AxisX, 3);
        Assert.Equal(5f, r.HalfWidth, 3);
    }

    [Fact]
    public void MinAreaRect_RotatedLine_FindsAngle()
    {
        // 30°-ban dőlt, 100 × 10-es téglalap sarkai
        float c = MathF.Cos(MathF.PI / 6), s = MathF.Sin(MathF.PI / 6);
        (float, float) P(float u, float v) => (u * c - v * s, u * s + v * c);
        var hull = DbPostProcessor.ConvexHull(new() { P(0, 0), P(100, 0), P(100, 10), P(0, 10) });
        var r = DbPostProcessor.MinAreaRect(hull);
        Assert.Equal(c, r.AxisX, 3);
        Assert.Equal(s, r.AxisY, 3);
        Assert.Equal(50f, r.HalfWidth, 2);
    }

    [Fact]
    public void Quad_SliceAlongWidth_ReturnsSubRectangle()
    {
        var q = new Quad(new(0, 0), new(100, 0), new(100, 20), new(0, 20));
        var half = q.SliceAlongWidth(0.25f, 0.75f);
        Assert.Equal(25f, half.Left, 3);
        Assert.Equal(75f, half.Right, 3);
        Assert.Equal(20f, half.Height, 3);
    }

    [Fact]
    public void PostProcessor_FindsTwoSeparateBlobs()
    {
        int w = 64, h = 32;
        var prob = new float[w * h];
        void Fill(int x0, int y0, int x1, int y1)
        {
            for (int y = y0; y < y1; y++)
                for (int x = x0; x < x1; x++)
                    prob[y * w + x] = 0.9f;
        }
        Fill(2, 2, 30, 8);
        Fill(2, 20, 50, 26);

        var boxes = new DbPostProcessor(0.3f, 0.6f, 1.5f).Process(prob, w, h);
        Assert.Equal(2, boxes.Count);
        Assert.All(boxes, b => Assert.True(b.HalfWidth > b.HalfHeight));
    }
}
