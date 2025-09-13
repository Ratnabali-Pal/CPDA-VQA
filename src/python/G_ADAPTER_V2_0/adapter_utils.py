

---

### 3. adapter_utils.py

This file contains all the helper functions for performing OCR, augmenting questions, generating image descriptions, and calculating the BLEU score.

---

**File Name:** `adapter_utils.py`
```python
# adapter_utils.py

import cv2
import numpy as np
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def m_augment(sentence, model, tokenizer, num_return_sequences=5, num_beams=5):
    """
    Generates paraphrased versions of a sentence using a Pegasus model.
    """
    inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
    outputs = model.generate(
        **inputs,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def m_description(image_path, model, image_processor, tokenizer, greedy=True):
    """
    Generates a caption for a given image.
    """
    image = Image.open(image_path)
    pixel_values = image_processor(image, return_tensors="pt").pixel_values

    if greedy:
        generated_ids = model.generate(pixel_values, max_new_tokens=30)
    else:
        generated_ids = model.generate(
            pixel_values,
            do_sample=True,
            max_new_tokens=30,
            top_k=5
        )
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def m_qa(image_path, augmented_questions, vqa_pipeline):
    """
    Generates answers for a list of augmented questions based on an image.
    """
    aqa_answers = []
    print("--- Running VQA on Augmented Questions ---")
    for q in augmented_questions:
        try:
            answer = vqa_pipeline(image_path, q, top_k=1)
            print(f"Q: {q} -> A: {answer[0]['answer']}")
            aqa_answers.append(answer[0]['answer'])
        except Exception as e:
            print(f"Could not answer question '{q}'. Error: {e}")
            
    return ", ".join(list(set(aqa_answers)))

def m_ocr(image_path, ocr_model):
    """
    Performs OCR on an image and returns the concatenated text.
    """
    result = ocr_model.ocr(image_path)
    concat_output = ""
    if result and result[0] is not None and np.size(result[0]) > 1:
        # The result from paddleocr can be nested. Ensure we extract text correctly.
        text_lines = [line[1][0] for line in result[0]]
        concat_output = ". ".join(text_lines)
    return concat_output

def calculate_bleu(answer, gt_answer):
    """
    Calculates BLEU-1 to BLEU-4 scores.
    """
    candidate = answer.lower().split()
    reference = [gt_answer.lower().split()]
    # Using SmoothingFunction to avoid zero scores for short sentences
    chencherry = SmoothingFunction()

    bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)
    bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
    bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1)
    bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)

    return [bleu1, bleu2, bleu3, bleu4]