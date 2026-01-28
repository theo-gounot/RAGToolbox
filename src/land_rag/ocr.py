import os
import base64
import requests
import subprocess
import tempfile
from pathlib import Path
from abc import abstractmethod
from typing import Any, List

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

# Conditional imports for heavy libraries to avoid import errors if dependencies aren't installed yet
try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
except ImportError:
    ocr_predictor = None
    DocumentFile = None

from src.land_rag.base import AbstractBaseModule
from src.land_rag.config import RAGConfig, OLLAMA_BASE_URL, VLM_MODEL

class OCRBase(AbstractBaseModule):
    @abstractmethod
    def run(self, file_path: str, **kwargs) -> str:
        """Extract text from a PDF file."""
        pass

    def _clean_to_markdown(self, text: str) -> str:
        """Uses LLM to clean raw OCR text into structured Markdown."""
        prompt = (
            "Clean and format the following raw OCR text into well-structured Markdown. "
            "Maintain headers, lists, and tables if present. Fix broken words or layout issues:\n\n"
            f"{text[:8000]}" # Limit to avoid context overflow
        )
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.config.ollama_model if hasattr(self.config, 'ollama_model') else "llama3.1:8b",
                    "prompt": prompt,
                    "stream": False
                }
            )
            if response.status_code == 200:
                return response.json().get("response", text)
        except Exception:
            pass
        return text

class LowOCR(OCRBase):
    """Extraction using pypdfium2 (Text only, fast)."""
    def run(self, file_path: str, **kwargs) -> str:
        pdf = pdfium.PdfDocument(file_path)
        text_parts = []
        for i in range(len(pdf)):
            page = pdf.get_page(i)
            text_page = page.get_textpage()
            text_parts.append(text_page.get_text_range())
        raw_text = "\n\n".join(text_parts)
        return self._clean_to_markdown(raw_text)

class VLMOCR(OCRBase):
    """Extraction by converting pages to images and sending to a remote VLM (Vision Model)."""
    def run(self, file_path: str, **kwargs) -> str:
        pdf = pdfium.PdfDocument(file_path)
        full_markdown = []
        
        for i in range(len(pdf)):
            page = pdf.get_page(i)
            # Render page to bitmap (using default scale 2 for decent resolution)
            bitmap = page.render(scale=2)
            pil_image = bitmap.to_pil()
            
            # Save to temp or convert to base64
            from io import BytesIO
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Send to Ollama
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": VLM_MODEL,
                    "prompt": "Extract the text from this image and format it as Markdown. Maintain tables and headers.",
                    "images": [img_str],
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                full_markdown.append(response.json().get("response", ""))
            else:
                full_markdown.append(f"Error processing page {i}: {response.text}")
                
        return "\n\n".join(full_markdown)

class MidOCR(OCRBase):
    """
    Extraction using DocTR (CPU) for layout and table detection.
    Ideal for documents with complex layouts but standard fonts.
    """
    def __init__(self, config: RAGConfig):
        super().__init__(config)
        if ocr_predictor is None:
            raise ImportError("python-doctr is not installed. Please install it to use MidOCR.")
        # Initialize model on CPU
        # det_arch: detection architecture (db_resnet50 is standard)
        # reco_arch: recognition architecture (crnn_vgg16_bn is robust)
        self.model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

    def run(self, file_path: str, **kwargs) -> str:
        # Load document
        doc = DocumentFile.from_pdf(file_path)
        # Run inference
        result = self.model(doc)
        # Export as string (DocTR's render method gives a decent visual representation)
        raw_text = result.render()
        return self._clean_to_markdown(raw_text)

class HighOCR(OCRBase):
    """
    Extraction using Nougat (via subprocess).
    Ideal for scientific papers and formulas.
    """
    def run(self, file_path: str, **kwargs) -> str:
        # We use a temporary directory to store the Nougat output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Construct command: nougat <input> -o <output_dir> --no-skipping
            # --no-skipping ensures it processes all pages
            cmd = [
                "nougat",
                file_path,
                "-o", temp_dir,
                "--no-skipping"
            ]
            
            try:
                # Run subprocess
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                # Nougat outputs a .mmd (Mathpix Markdown) file with the same stem as input
                input_path = Path(file_path)
                output_file = Path(temp_dir) / f"{input_path.stem}.mmd"
                
                if output_file.exists():
                    return output_file.read_text(encoding='utf-8')
                else:
                    raise FileNotFoundError(f"Nougat output file not found: {output_file}")
                    
            except subprocess.CalledProcessError as e:
                # Handle execution errors
                raise RuntimeError(f"Nougat OCR failed: {e.stderr}")

class OCRFactory:
    @staticmethod
    def get_ocr(config: RAGConfig) -> OCRBase:
        method = config.ocr_method
        if method == "low":
            return LowOCR(config)
        elif method == "VLM":
            return VLMOCR(config)
        elif method == "mid":
            return MidOCR(config)
        elif method == "high":
            return HighOCR(config)
        else:
            raise ValueError(f"Unknown OCR method: {method}")