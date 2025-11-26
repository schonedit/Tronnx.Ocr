# Tronnx.Ocr – Fully Offline OCR Engine for .NET

Tronnx.Ocr is a fully offline OCR library for .NET.  
It performs text detection and text recognition using ONNX Runtime and OpenCV, powered by two Apache 2.0 licensed ONNX models.  
All models are embedded directly into the NuGet package, so no external downloads or network services are required.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Included ONNX Models](#included-onnx-models)
- [How It Works](#how-it-works)
- [Privacy](#privacy)
- [Dependencies & Licenses](#dependencies--licenses)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Fully offline OCR pipeline (no network usage, no telemetry)
- Fast inference based on ONNX Runtime
- OpenCV-based preprocessing and region extraction
- Embedded ONNX models (no separate installation needed)
- Automatic model caching for faster repeated runs
- Multi-line recognition with reading-order sorting
- Simple and developer-friendly API

---

## Installation

Using **NuGet Package Manager**:

```powershell
Install-Package Tronnx.Ocr
```

Using **.NET CLI**:

```bash
dotnet add package Tronnx.Ocr
```

---

## Quick Start

```csharp
using Tronnx.Ocr;

var ocr = new TronnxOCR();

// 1) Extract text
string text = ocr.GetText("image.png");

// 2) Generate searchable PDF
ocr.ToSearchablePdf(new[] { "image1.png", "image2.png" }, "output.pdf");

// 3) Extract text + generate PDF in one call
string text = ocr.GetTextAndPdf(new[] { "image.png" }, "output.pdf");
```

---

## Included ONNX Models

The package bundles the following Apache 2.0 licensed ONNX models:

| Purpose          | Model Name                | License    |
|------------------|---------------------------|------------|
| Text Detection   | linknet_resnet18.onnx     | Apache 2.0 |
| Text Recognition | crnn_vgg16_bn_dynW.onnx   | Apache 2.0 |

The models are embedded resources and are automatically extracted into a temporary cache folder on first use.

---

## How It Works

1. Load the input image  
2. Preprocess (resize & pad)  
3. Detect text regions using the detector model  
4. Remap boxes to the original image scale  
5. Sort detected regions in reading order  
6. Recognize text lines using the recognition model  
7. Return the combined OCR result  

---

## Privacy

Tronnx.Ocr operates entirely locally:

- No external services  
- No cloud processing  
- No telemetry  
- No hidden network dependencies  
- All computation happens on the user’s machine  

Your images and text never leave the device.

---

## Dependencies & Licenses

This project uses the following open-source libraries:

- Microsoft.ML.OnnxRuntime – MIT License  
- OpenCvSharp4 – MIT License  
- OpenCvSharp4.runtime.win – MIT License  
- PdfSharpCore (MIT License)
- SkiaSharp (MIT License)

Bundled ONNX models are licensed under Apache License 2.0.

---

## Contributing

This library is in an early development phase and will continue to evolve.  
Suggestions, issues, and pull requests are welcome on the GitHub repository.

---

## License

The Tronnx.Ocr library is released under the MIT License.  
Bundled ONNX models are provided under the Apache 2.0 License.


