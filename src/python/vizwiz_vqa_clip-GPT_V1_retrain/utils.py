# utils.py

import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def find_most_common_answer(answers):
    """Finds the most common answer from a list of answers."""
    answer_counter = Counter(answers)
    most_common_answers = answer_counter.most_common()
    most_common_answer, count = most_common_answers[0]
    return most_common_answer

def select_most_common_answers(df):
    """Selects the most common answer for each question in the DataFrame."""
    selected_answers = []
    for idx, row in df.iterrows():
        answers = [answer["answer"] for answer in row["answers"]]
        selected_answer = find_most_common_answer(answers)
        selected_answers.append({"answer": selected_answer})
    df[["answer"]] = pd.DataFrame(selected_answers)
    return df.drop(["answers"], axis=1)

def dataloader_json(path, test=False):
    """Loads and processes the JSON data from the given path."""
    with open(path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if test:
        return df
    return select_most_common_answers(df)

def plot_loss(train_loss, val_loss):
    """Plots the training and validation loss."""
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def plot_img(path):
    """Plots an image from the given path."""
    image = mpimg.imread(path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()