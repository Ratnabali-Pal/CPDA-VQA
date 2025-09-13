import torch
import clip
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from model import VQAModel
from data_loader import plot_img, dataloader_json

def encode(question, image_path, model_clip, preprocess, device):
    """Encodes a question and image for the model."""
    plot_img(image_path)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(question).to(device)
    
    with torch.no_grad():
        image_encoding = model_clip.encode_image(image)
        text_encoding = model_clip.encode_text(text)
    return torch.cat([image_encoding, text_encoding], dim=-1).float()

def predict(question, image_path, model, model_clip, preprocess, device, ans_lb, ans_type_lb):
    """Makes a prediction on a given question and image."""
    model.eval()
    with torch.no_grad():
        encoded_input = encode(question, image_path, model_clip, preprocess, device)
        output, aux = model(encoded_input)
        _, test_predicted = torch.max(output, dim=1)
        _, test_predicted_ty = torch.max(aux, dim=1)
        
        answer = ans_lb.inverse_transform([test_predicted.item()])[0]
        answer_type = ans_type_lb.inverse_transform([test_predicted_ty.item()])[0]
        
        print(f'Answer: {answer}, Answer Type: {answer_type}')

def main():
    # Load the label encoders
    # This assumes you have run the training script and have the dataset available
    try:
        train_df = dataloader_json("dataset/Annotations/Annotations/generated.json")
        val_df = dataloader_json("dataset/Annotations/Annotations/val.json")
    except FileNotFoundError:
        print("Error: Annotation files not found. Please make sure the dataset is in the correct directory structure.")
        return
        
    data_df = pd.concat((train_df, val_df), axis=0, ignore_index=True)

    ans_lb = LabelEncoder()
    ans_lb.fit(data_df['answer'])
    ans_type_lb = LabelEncoder()
    ans_type_lb.fit(data_df['answer_type'])

    embedding_size = 768
    num_classes = len(ans_lb.classes_)
    num_aux_classes = len(ans_type_lb.classes_)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = VQAModel(embedding_size, num_classes, num_aux_classes)
    
    # Load the trained model weights
    try:
        # If you used DataParallel, you need to load the state_dict carefully
        state_dict = torch.load('best_model.pt')
        if isinstance(model, torch.nn.DataParallel):
             model.module.load_state_dict(state_dict)
        else:
             # Create a new state_dict without the 'module.' prefix
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

    except FileNotFoundError:
        print("Error: 'best_model.pt' not found. Please train the model first by running train.py.")
        return
        
    model.to(device)

    try:
        model_clip, preprocess = clip.load("ViT-L/14", device=device)
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return

    # Example usage:
    path = "dataset/test/test/VizWiz_test_00000004.jpg"
    question = "What is this?"
    predict(question, path, model, model_clip, preprocess, device, ans_lb, ans_type_lb)

    path = "dataset/test/test/VizWiz_test_00000011.jpg"
    question = "What's the date?"
    predict(question, path, model, model_clip, preprocess, device, ans_lb, ans_type_lb)


if __name__ == "__main__":
    main()