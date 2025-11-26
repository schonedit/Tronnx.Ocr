using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tronnx.Ocr.Types
{
    internal class DetectedBox
    {
        public float X { get; set; }   
        public float Y { get; set; }      
        public float Width { get; set; }  
        public float Height { get; set; } 
        public string Text { get; set; } = string.Empty;
    }
}
