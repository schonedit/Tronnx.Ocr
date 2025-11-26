using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tronnx.Ocr.Types
{
    internal class PageResult
    {
        public string ImagePath { get; set; } = string.Empty;
        public int ImageWidth { get; set; }
        public int ImageHeight { get; set; }
        public List<DetectedBox> Boxes { get; set; } = new();
    }
}
