# adopted from https://github.com/huggingface/datasets/blob/main/metrics/rouge/rouge.py


from rouge_score import rouge_scorer
# from multiprocessing import Pool


def rouge_l(predictions, references, use_stemmer=False):
    scorer = rouge_scorer.RougeScorer(rouge_types=['rougeL'], use_stemmer=use_stemmer)
    scores = []
    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        scores.append(score)

    rouge_l_score = scores[0]['rougeL'].fmeasure
    return rouge_l_score
