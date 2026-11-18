# Letölti a PP-OCRv5 ONNX modelleket a Tronnx.Ocr.Models.Latin projekt models/ mappájába,
# és SHA-256 ellenőrzőösszeggel ellenőrzi őket. Futtatás a repó gyökeréből:
#   pwsh tools/download-models.ps1
# Javaslat: töltsd fel ugyanezeket a saját GitHub-repód "models-v1" release-ébe, és írd át a $BaseUrl-t.
$ErrorActionPreference = 'Stop'

$BaseUrl = if ($env:TRONNX_MODELS_URL) { $env:TRONNX_MODELS_URL } else { 'https://github.com/GreatV/oar-ocr/releases/download/v0.3.0' }
$Dest = Join-Path $PSScriptRoot '..\src\Tronnx.Ocr.Models.Latin\models'
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

$Files = @{
    'pp-ocrv5_mobile_det.onnx'       = '1eb7b4f7ab657ebd1c66d5f79bca7497f29768a2e3c15e52daecbba1a8e4a039'
    'latin_pp-ocrv5_mobile_rec.onnx' = 'e3a6bfeea1c8a01d6fccfd480a0bd363fd907f8c65931e228bb2736f5c3e142f'
    'ppocrv5_latin_dict.txt'         = 'ccbcc45730b3fbbd9050c5bc74db6a99067141ef1035e3d14889a84a6b9b1aff'
}

foreach ($name in $Files.Keys) {
    $target = Join-Path $Dest $name
    if (-not (Test-Path $target)) {
        Write-Host "Downloading $name ..."
        Invoke-WebRequest -Uri "$BaseUrl/$name" -OutFile $target
    }
    $actual = (Get-FileHash -Algorithm SHA256 $target).Hash.ToLowerInvariant()
    if ($actual -ne $Files[$name]) {
        Remove-Item $target
        throw "Checksum mismatch for $name (got $actual)"
    }
    Write-Host "OK  $name"
}
