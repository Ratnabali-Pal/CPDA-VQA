# main.py

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from models import load_all_models
from adapter_utils import (
    m_ocr,
    m_augment,
    m_qa,
    m_description,
    calculate_bleu
)

def g_adapter(image_path, question, models):
    """
    The main G-ADAPTER pipeline function.
    """
    # Step 1: Extract text from the image using OCR
    print("\n--- Step 1: Performing OCR ---")
    ocr_text = m_ocr(image_path, models["ocr"])
    print(f"OCR Output: {ocr_text}")

    # Step 2: Generate paraphrased questions
    print("\n--- Step 2: Augmenting Question ---")
    augmented_questions = m_augment(
        question, 
        models["augment_model"], 
        models["augment_tokenizer"]
    )
    print(f"Augmented Questions: {augmented_questions}")

    # Step 3: Answer augmented questions using the VQA model
    print("\n--- Step 3: Answering Augmented Questions ---")
    qa_answers = m_qa(image_path, augmented_questions, models["vqa"])
    print(f"Consolidated VQA Answers: {qa_answers}")

    # Step 4: Generate a description of the image
    print("\n--- Step 4: Generating Image Description ---")
    description_text = m_description(
        image_path,
        models["description_model"],
        models["image_processor"],
        models["description_tokenizer"]
    )
    print(f"Image Description: {description_text}")

    # Step 5: Combine all information to create a context and get the final answer
    print("\n--- Step 5: Synthesizing Final Answer ---")
    context = f"Image description: {description_text}. Text found in image: {ocr_text}. Related answers: {qa_answers}."
    
    llm_output = models["llm_qa"](question=question, context=context)
    final_answer = llm_output['answer']

    print("\n--- G-ADAPTER FINAL OUTPUT ---")
    print(f"Generated Context: {context}")
    print(f"Original Question: {question}")
    print(f"Final Answer: {final_answer}")
    
    return final_answer

if __name__ == "__main__":
    # --- Configuration ---
    # Update these paths and questions for your use case.
    # Ensure the image path is correct. You might need to create a 'data' folder.
    IMAGE_PATH = "VizWiz_train_00000016.jpg" 
    QUESTION = "What is this?"
    GROUND_TRUTH_ANSWER = "Tea"

    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        print("Please update the IMAGE_PATH variable in main.py")
    else:
        # Load all models
        loaded_models = load_all_models()

        # Display the image
        image = Image.open(IMAGE_PATH).convert("RGB")
        plt.imshow(np.asarray(image))
        plt.title("Input Image")
        plt.show()

        # Run the G-ADAPTER pipeline
        final_answer = g_adapter(IMAGE_PATH, QUESTION, loaded_models)

        # Calculate BLEU score
        bleu_scores = calculate_bleu(final_answer, GROUND_TRUTH_ANSWER)
        
        print("\n--- Evaluation ---")
        print(f"Ground Truth: {GROUND_TRUTH_ANSWER}")
        print(f"BLEU-1 Score: {bleu_scores[0]:.4f}")
        print(f"BLEU-2 Score: {bleu_scores[1]:.4f}")
        print(f"BLEU-3 Score: {bleu_scores[2]:.4f}")
        print(f"BLEU-4 Score: {bleu_scores[3]:.4f}")