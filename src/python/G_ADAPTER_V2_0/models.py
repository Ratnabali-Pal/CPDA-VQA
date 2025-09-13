# models.py

from transformers import (
    pipeline,
    PegasusForConditionalGeneration,
    PegasusTokenizerFast,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    GPT2TokenizerFast
)
from paddleocr import PaddleOCR
import warnings

warnings.filterwarnings('ignore')

def load_all_models():
    """
    Loads and initializes all the models required for the G-Adapter pipeline.
    """
    print("Loading all models. This may take a while...")

    # 1. Visual Question Answering Model
    vqa_baseline = pipeline("visual-question-answering", model="MariaK/vilt_finetuned_200")

    # 2. Text Augmentation/Paraphrasing Model
    augment_model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
    augment_tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")

    # 3. Image Captioning Model
    description_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    description_tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    # 4. OCR Model
    ocr_baseline = PaddleOCR(use_angle_cls=True, lang='en')

    # 5. LLM for final Question Answering
    llm_qa_baseline = pipeline('question-answering')

    print("All models loaded successfully.")

    models = {
        "vqa": vqa_baseline,
        "augment_model": augment_model,
        "augment_tokenizer": augment_tokenizer,
        "description_model": description_model,
        "image_processor": image_processor,
        "description_tokenizer": description_tokenizer,
        "ocr": ocr_baseline,
        "llm_qa": llm_qa_baseline
    }
    
    return models