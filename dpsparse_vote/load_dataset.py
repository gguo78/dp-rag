# load_dataset.py

from datasets import load_dataset

def load_triviaqa_subset(split='validation', num_samples=100):
    """
    Load a small subset of TriviaQA for testing.
    """
    dataset = load_dataset("trivia_qa", "rc.nocontext", split=split)
    questions = []
    answers = []

    for item in dataset.select(range(num_samples)):
        questions.append(item["question"])
        # Use first answer for simplicity
        if isinstance(item["answer"]["value"], list):
            answers.append(item["answer"]["value"][0])
        else:
            answers.append(item["answer"]["value"])

    return questions, answers

def load_natural_questions_subset(split='validation', num_samples=100):
    """
    Load a small subset of NaturalQuestions for testing.
    """
    dataset = load_dataset("natural_questions", split=split)
    questions = []
    answers = []

    for item in dataset.select(range(num_samples)):
        questions.append(item["question"])
        answers.append(item.get("answer", ""))  # Some answers might be missing

    return questions, answers
