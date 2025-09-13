import os
from data_utils import VizWizOCRDataset
from inference import run_ocr_inference
from utils import show_image

if __name__ == "__main__":
    image_path = "sample.jpg"  # Replace with actual image
    question = "What text is written on this label?"

    print("Running OCR on:", image_path)
    result = run_ocr_inference(image_path, question)
    print(result)

    show_image(image_path, title="OCR Example")
