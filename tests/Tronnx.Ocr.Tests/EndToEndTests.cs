using Tronnx.Ocr.Models;
using Xunit;

namespace Tronnx.Ocr.Tests;

/// <summary>Valódi modellekkel futó teszt. A motor betöltése drága, ezért osztályonként egyszer.</summary>
public sealed class EngineFixture : IDisposable
{
    public OcrEngine Engine { get; } = OcrEngine.Create(new LatinModelPack());
    public void Dispose() => Engine.Dispose();
}

public class EndToEndTests : IClassFixture<EngineFixture>
{
    private readonly OcrEngine _engine;

    public EndToEndTests(EngineFixture fixture) => _engine = fixture.Engine;

    [Theory]
    [InlineData("assets/hu_test.png")]
    [InlineData("assets/hu_test_rot.png")] // 4°-kal elforgatva
    public void ReadsHungarianText(string path)
    {
        var page = _engine.Recognize(path);

        Assert.Equal(5, page.Lines.Count);
        Assert.Contains("Tisztelt Ügyfelünk! Értesítjük, hogy a szerződése", page.Text);
        Assert.Contains("Fizetendő összeg: 12 345 Ft.", page.Text);
        Assert.All(page.Lines, l => Assert.True(l.Confidence > 0.9f));
    }

    [Fact]
    public void ProducesWordBoxesInsideLineBoxes()
    {
        var page = _engine.Recognize("assets/hu_test.png");
        var line = page.Lines[1];

        Assert.Equal(line.Text.Split(' ').Length, line.Words.Count);
        foreach (var w in line.Words)
        {
            Assert.True(w.Box.Left >= line.Box.Left - 1 && w.Box.Right <= line.Box.Right + 1);
        }
        // a szavak balról jobbra követik egymást
        for (int i = 1; i < line.Words.Count; i++)
            Assert.True(line.Words[i].Box.Left > line.Words[i - 1].Box.Left);
    }
}
