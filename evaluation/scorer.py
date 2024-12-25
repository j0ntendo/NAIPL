from nltk.translate.bleu_score import sentence_bleu

def evaluate_bleu(pred, ref):
    return sentence_bleu([ref.split()], pred.split())