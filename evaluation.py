import torch
import numpy as np
import sys

from sklearn.metrics import cohen_kappa_score as kappa, roc_auc_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


def predict_dataset(dataset, model, use_cuda=True, bs=32):
    model.eval()

    X = dataset.X
    Y = dataset.Y
    Yhat = []

    iSample = 0
    while iSample < X.shape[0]:
        print(str(iSample + 1) + '/' + str(X.shape[0]), end='\r')

        batch = X[iSample:iSample+bs, :, :, :]
        batch = torch.tensor(batch).float()
        if use_cuda:
            batch = batch.cuda()

        # Forward
        yhat = model(batch)

        Yhat.append(yhat.detach().cpu().numpy())

        iSample += bs

    Yhat = np.concatenate(Yhat, 0)
    return Y, Yhat


def eval_predictions_multi(y_true, y_pred, print_conf=True, classes=['NC', 'G3', 'G4', 'G5']):
    acc = balanced_accuracy_score(y_true, y_pred)
    k = kappa(y_true, y_pred, weights='quadratic')

    if print_conf:
        cm = confusion_matrix(y_true, y_pred, labels=list(np.arange(0, len(classes))))
        print_cm(cm, classes)

    return k, acc


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """
    pretty print for confusion matrixes
    https://gist.github.com/zachguo/10296432
    """
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()
