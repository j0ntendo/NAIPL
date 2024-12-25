# scorer.py
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate_bleu(pred, ref):
    smoothie = SmoothingFunction().method1()
    return sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)