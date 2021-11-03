from typing import Dict
from .configs import EvalConfig
from collections import Counter
from sklearn.metrics import confusion_matrix
import pet.trainers.meters as meters
import numpy as np
import log

logger = log.get_logger("root")


def singleton_evaluate(results: Dict, config: EvalConfig,) -> Dict:
    metrics = config.metrics if config.metrics else ["acc"]
    predictions = np.argmax(results["logits"], axis=1)
    guids = results["guids"]
    golds = results["labels"]
    scores = {}
    for metric in metrics:
        if metric == "acc":
            scores[metric] = meters.accuracy(results["labels"], predictions)
        elif metric == "f1":
            scores[metric] = meters.f1_score(results["labels"], predictions)
        elif metric == "f1-macro":
            scores[metric] = meters.macro_f1_(
                results["labels"], predictions, average="macro"
            )
        elif metric == "em":
            scores[metric] = meters.exact_match(
                predictions, results["labels"], results["question_ids"]
            )
        else:
            raise ValueError(f"Metric '{metric}' not implemented")
    logger.info(f"Gold label distribution: {Counter(results['labels'])}")
    logger.info(f"Predicted distribution: {Counter(predictions)}")
    logger.info(f"Confusion matrix {confusion_matrix(results['labels'], predictions)}")

    assert len(predictions) == len(guids) == len(golds)

    results["scores"] = scores
    results["predictions"] = predictions
    return (results, list(zip(predictions.tolist(), golds.tolist(), guids)))
