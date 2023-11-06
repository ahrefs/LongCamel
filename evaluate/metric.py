from rouge_score import rouge_scorer
from multiprocessing import Pool
import numpy as np


def rouge(predictions, references, rouge_types=None, use_stemmer=False, return_geometric_mean=True):
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
    with Pool() as p:
        scores = p.starmap(scorer.score, zip(references, predictions))
    if return_geometric_mean:
        return np.mean([np.exp(np.mean(np.log([s.fmeasure for s in score.values()]))) for score in scores])
    result = {}
    for key in scores[0]:
        result[key] = list(score[key] for score in scores)
    return result




