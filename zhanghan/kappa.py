# kappa for kaggle contest ()
import numpy as np



def expected_matrices(target):
    w, h = 8, 8
    Matrix = [[0 for x in range(w)] for y in range(h)]
    counted = np.bincount(target)[1:]
    for i in range(8):
        Matrix[i][i] = counted[i]
    return Matrix

def estimate_matrices(predict, y_test):
    w, h = 8, 8
    Matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(8):
        index_tmp = y_test[y_test == i + 1].index
        for j in range(8):
            impute = sum(predict[index_tmp] == j+1)
            Matrix[i][j] = impute
    return Matrix

def weighted_matrices(expected, estimate):
    n = 8
    w, h = 8, 8
    Matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(8):
        for j in range(8):
            Matrix[i][j] = ((estimate[i][j] - expected[i][j])**2)/((n-1)**2)

    return Matrix

def kappa(y_test, y_predict):
    expected = expected_matrices(y_test)
    estimate = estimate_matrices(y_predict, y_test)
    weighted = weighted_matrices(expected, estimate)

    numerator = sum(sum(np.array(estimate) * np.array(weighted)))
    denominator = sum(sum(np.array(expected) * np.array(weighted)))
    kappa = 1 - numerator/denominator

    return kappa