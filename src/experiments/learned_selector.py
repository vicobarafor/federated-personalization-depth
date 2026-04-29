
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score


LABEL_MAP = {
    "head_only": 0,
    "conv2_head": 1,
    "full_ft": 2,
}

INV_LABEL_MAP = {v:k for k,v in LABEL_MAP.items()}


def oracle_label(result_dict):
    scores = {
        "head_only": result_dict["head_only"]["accuracy"],
        "conv2_head": result_dict["conv2_head"]["accuracy"],
        "full_ft": result_dict["full_ft"]["accuracy"],
    }
    return max(scores.items(), key=lambda x:x[1])[0]


def train_loocv_selector(X, y):
    loo = LeaveOneOut()

    preds = []
    truth = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            random_state=0
        )

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)[0]

        preds.append(pred)
        truth.append(y_test[0])

    acc = accuracy_score(truth, preds)

    return acc, preds, truth


def fit_full_selector(X, y):
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        random_state=0
    )
    clf.fit(X, y)
    return clf
