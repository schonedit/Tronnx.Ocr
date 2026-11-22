# Tronnx.Ocr

Fast, fully offline OCR for .NET, built on ONNX Runtime and the PP-OCRv5 models.

- **Offline and private.** Inference runs locally. There is no network access and no telemetry.
- **Cross-platform.** Runs on Windows, Linux (including Docker) and macOS. It has no native OpenCV dependency.
- **32 Latin-script languages**, including English, German, French, Spanish, Polish, Czech and Hungarian.
- **Detailed results.** Each line and word comes with a rotated bounding box and a confidence score.
- Searchable PDF output and word highlighting will return in 2.0. *(This is a preview; see the roadmap below.)*

> **2.0 preview:** This is a rewrite of the 1.x proof of concept. The 1.x `TronnxOCR.GetText()` API is kept for compatibility.

## Install

```
dotnet add package Tronnx.Ocr --prerelease
```

## Usage

```csharp
using Tronnx.Ocr;
using Tronnx.Ocr.Models;

// Create the engine once and reuse it. It is thread-safe.
using var engine = OcrEngine.Create(new LatinModelPack());

OcrPage page = engine.Recognize("scan.png");
Console.WriteLine(page.Text);

foreach (var line in page.Lines)
    Console.WriteLine($"{line.Confidence:P0} {line.Text} @ {line.Box.Left},{line.Box.Top}");
```

1.x-style code still works:

```csharp
using var ocr = new TronnxOCR();
string text = ocr.GetText("scan.png");
```

### Options

```csharp
var options = new OcrEngineOptions
{
    DetectionMaxSideLength = 1280, // raise this for very small print
    IntraOpThreads = 4,
    MinLineConfidence = 0.6f,
};
```

## Building from source

```
pwsh tools/download-models.ps1      # or: bash tools/download-models.sh
dotnet build
dotnet test
dotnet run --project samples/Tronnx.Ocr.Cli -- path/to/image.png --bench 10
```

## Roadmap

- [x] PP-OCRv5 detection and Latin recognition with pure-C# geometry (no OpenCV)
- [x] Word-level boxes
- [ ] Searchable PDF and highlighting (`Tronnx.Ocr.Pdf`)
- [ ] PDF input (OCR of scanned PDFs)
- [ ] **Tronnx.Ocr.Pro** (commercial): a fine-tuned Hungarian model with higher accuracy on ő/ű, GPU acceleration, more European language packs, and priority support

## License

[Apache License 2.0](LICENSE). The bundled PaddleOCR models are also Apache 2.0; see [NOTICE](NOTICE).
The Tronnx.Ocr.Pro add-on will be sold under a separate commercial license.
