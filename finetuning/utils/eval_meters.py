from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
)
from collections import Counter


def show_confusion(y, pred):
    print(f"Gold distribution {Counter(y)}")
    print(f"Pred distribution {Counter(pred)}")
    print(confusion_matrix(y, pred))


def accuracy(y, pred):
    return accuracy_score(y, pred)


def f1(y, pred, average=None):
    if average:
        return f1_score(y, pred, average=average)
    return f1_score(y, pred)


def recall(y, pred, average=None):
    if average:
        return recall_score(y, pred, average=average)
    return recall_score(y, pred)


def precision(y, pred, average=None):
    if average:
        return precision_score(y, pred, average=average)
    return precision_score(y, pred)


def micro_recall(y, pred):
    return recall(y, pred, average="micro")


def micro_precision(y, pred):
    return precision(y, pred, average="micro")


def micro_f1(y, pred):
    return f1(y, pred, average="micro")


def mcc(y, pred):
    return matthews_corrcoef(y, pred)
