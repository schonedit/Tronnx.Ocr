using OpenCvSharp;


namespace Tronnx.Ocr.Types
{
    internal class BoxPoint
    {
        public Point TopLeft { get; set; }
        public Point TopRight { get; set; }
        public Point BottomLeft { get; set; }
        public Point BottomRight { get; set; }
        public int BoxIndex { get; set; }
        public RotatedRect Box { get; set; }
    }
}
