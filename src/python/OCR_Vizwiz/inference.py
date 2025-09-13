import torch
from PIL import Image
from ocr_model import OCRModel

def run_ocr_inference(image_path, question=None):
    """
    Runs OCR on the image and returns extracted text.
    If a question is provided, it combines OCR with question.
    """
    ocr = OCRModel()
    extracted_text = ocr.extract_text(image_path)

    if question:
        return f"Q: {question}\nOCR Result: {extracted_text}"
    return extracted_text
