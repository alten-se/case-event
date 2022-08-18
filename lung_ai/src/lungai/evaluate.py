from typing import Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, classification_report,
    f1_score, matthews_corrcoef, cohen_kappa_score
)
import numpy as np
from numpy import ndarray
from keras.models import Sequential
from soundfile import SoundFile

from lungai.data_extraction import extract_mfccs

def eval_sound(file, model, label_dict):
    s_file = SoundFile(file, "r")
    mffcs = extract_mfccs(s_file)
    return eval(mffcs.T, model, label_dict)
    

def invert_dict(k_v: dict) -> dict:
    """Creates a new dict that is the inverted input dict.

    Args:
        k_v (dict): a dict{ keys, values}

    Returns:
        dict: v_k
    """
    return {v: k for k, v in k_v.items()}
 
def eval(input: ndarray, model: Sequential, label_dict: dict) -> Tuple[str, float]:
    """_summary_

    Args:
        input (ndarray): single data point
        model (Sequential): model to use for prediction
        label_dict (dict): mapping from str to int labels

    Returns: 
        Tuple(
            str: predicted str label,
            float: confidence level between 0 to 1.0
        )
    """
    inv_map = invert_dict(label_dict)

    pred = np.array(model(input[np.newaxis, :]))
    index = np.argmax(pred)

    return inv_map[index], pred[0, index]

def evalModel(y_test, y_pred):
    '''
        Evaluate the performance of the model.
        Args:
           y_test: The array of features to be tested against.
           y_pred: Model predictions.
    Returns: Accuracy, Precision, Recall, F1 score, Cohens kappa, Matthews correlation coefficient
             of the model after evaluation.

    '''
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])
    y_test = np.argmax(y_test, axis=1)

# accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, y_pred, average='weighted')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, y_pred, average='weighted')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print('F1 score: %f' % f1)

    # kappa
    kappa = cohen_kappa_score(y_test, y_pred)
    print('Cohens kappa: %f' % kappa)
    MatthewsCorrCoef = matthews_corrcoef(y_test, y_pred)
    print('Matthews correlation coefficient: %f' % MatthewsCorrCoef)
    # ROC AUC
    '''auc = roc_auc_score(y_test, y_pred)
    print('ROC AUC: %f' % auc)'''
    # confusion matrix
    matrix = classification_report(y_test, y_pred)
    print(matrix)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 score": f1,
        "Cohens kappa": kappa,
        "Matthews correlation coefficient": MatthewsCorrCoef
    }
