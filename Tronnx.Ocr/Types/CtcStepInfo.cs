namespace Tronnx.Ocr.Types
{
    internal class CtcStepInfo
    {
        public int Step { get; set; }
        public int BestClass { get; set; }
        public float BestLogit { get; set; }
        public float SecondBestLogit { get; set; }
        public float MaxProb { get; set; }
        public float Entropy { get; set; }
        public bool IsBlank { get; set; }
        public char? Character { get; set; }
    }
}
