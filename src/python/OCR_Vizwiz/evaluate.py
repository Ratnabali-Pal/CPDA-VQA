from collections import Counter

def find_most_common_answer(answers):
    answer_counter = Counter(answers)
    most_common_answers = answer_counter.most_common()
    most_common_answer, _ = most_common_answers[0]
    return most_common_answer

def compute_accuracy(preds, labels):
    correct = sum([p == l for p, l in zip(preds, labels)])
    return correct / len(labels)
