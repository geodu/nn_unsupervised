import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

use_svm = True
def _get_classifier(c):
    if use_svm:
        return SVC(C=c, kernel='rbf', gamma=0.0005)
    else:
        return LogisticRegression(C=c)

def get_nn_features():
    test_dict = np.load('ddsm_test.npz')
    train_dict = np.load('ddsm_train.npz')

    train_val_feature_vectors = train_dict['data']
    train_val_labels = train_dict['labels']
    test_feature_vectors = test_dict['data']
    test_labels = test_dict['labels']

    scaler = StandardScaler().fit(train_val_feature_vectors)
    train_val_feature_vectors = scaler.transform(train_val_feature_vectors)
    test_feature_vectors = scaler.transform(test_feature_vectors)

    total_train_val = len(train_val_labels)
    total_val = total_train_val // 5

    train_feature_vectors = train_val_feature_vectors[total_val:, :]
    validation_feature_vectors = train_val_feature_vectors[:total_val, :]
    train_labels = train_val_labels[total_val:]
    validation_labels = train_val_labels[:total_val]

    """
    test_feature_vectors = sparse.csr_matrix(test_feature_vectors)
    train_val_feature_vectors = sparse.csr_matrix(train_val_feature_vectors)
    train_feature_vectors = sparse.csr_matrix(train_feature_vectors)
    validation_feature_vectors = sparse.csr_matrix(validation_feature_vectors)
    """

    print("Split dataset into training, test and validation sets.")
    print("Training size: {}".format(np.shape(train_feature_vectors)))
    print("Validation size: {}".format(np.shape(validation_feature_vectors)))
    print("Test size: {}".format(np.shape(test_feature_vectors)))
    return train_feature_vectors, validation_feature_vectors, train_val_feature_vectors, test_feature_vectors, train_labels, validation_labels, train_val_labels, test_labels

def check_matches(labels, predicted_labels):
    if np.size(labels) != np.size(predicted_labels):
        raise ValueError
    num_matches = np.sum(np.array(labels) == np.array(predicted_labels))
    proportion_matched = num_matches / np.size(labels)
    return proportion_matched

def evaluate_pairs():
    process(*get_nn_features())

def process(train_feature_vectors, validation_feature_vectors, train_val_feature_vectors, test_feature_vectors, train_labels, validation_labels, train_val_labels, test_labels):
    print("Using validation set to optimize over value of regularization parameter in regression, C.")
    best_proportion_matched = 0
    C_RANGE = [1, 10, 100, 1000, 10000, 100000]
    for C_cur in C_RANGE:
        """
        It turns out that the default value is pretty good, with performance smoothly increasing then smoothly
        decreasing after 1. You can choose to verify this by passing in
        C_RANGE = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]
        """
        print("\tCurrent C: {}".format(C_cur))
        classifier = _get_classifier(C_cur)
        classifier.fit(train_feature_vectors, train_labels)
        predicted_train_labels = classifier.predict(train_feature_vectors)
        predicted_validation_labels = classifier.predict(validation_feature_vectors)
        accuracy_train = check_matches(train_labels, predicted_train_labels)
        accuracy_validation = check_matches(validation_labels, predicted_validation_labels)
        print("\t\tTraining accuracy: {}".format(accuracy_train))
        print("\t\tValidation accuracy: {}".format(accuracy_validation))
        if accuracy_validation > best_proportion_matched:
            best_C = C_cur
            best_proportion_matched = accuracy_validation

    print("Done training and validating. Best C found: {}, Best accuracy on validation: {}".format(best_C, best_proportion_matched))

    classifier = _get_classifier(best_C)
    classifier.fit(train_val_feature_vectors, train_val_labels)
    predicted_train_labels = classifier.predict(train_feature_vectors)
    predicted_test_labels = classifier.predict(test_feature_vectors)
    accuracy_train = check_matches(train_labels, predicted_train_labels)
    accuracy_test = check_matches(test_labels, predicted_test_labels)
    print("Accuracy on train {}".format(accuracy_train))
    print("Accuracy on test {}".format(accuracy_test))

if __name__ == '__main__':
    evaluate_pairs()
