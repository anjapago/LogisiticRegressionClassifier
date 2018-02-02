import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LogisticRegression

# to be run on each class matrix, once for train and once for test
def format_data(mat, train=True):
    class1 = mat['class_1']
    class2 = mat['class_2']
    
    train_percent = 0.8
    train_size = int(train_percent*(class1.shape[1]))
    
    if(train):
        class1_features = class1.transpose()[0:train_size, :]
        class2_features = class2.transpose()[0:train_size, :]
    else:
        class1_features = class1.transpose()[train_size:, :]
        class2_features = class2.transpose()[train_size:, :]
    
    features = np.vstack((class1_features, class2_features))
    
    # add additional column b
    intercept = np.ones((features.shape[0], 1))
    features = np.hstack((intercept, features))

    # class ID list
    classIDs = [0]*class1_features.shape[0]
    for one in [1]*class2_features.shape[0]:
        classIDs.append(one)
    
    return features, classIDs

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll

def logistic_regression(data = 'classify_d5_k3_saved1.mat', learning_rate = 0.0001, num_steps = 50000):
    # load data
    mat = scipy.io.loadmat(data)
    features, classIDs = format_data(mat)

    # initialize param variable
    params = np.ones(features.shape[1])

	# gradient descent:
    #print("beginging gradient descent, log-likelihoods:")
    for step in range(0, num_steps):
        scores = np.dot(features, params)
        predictions = sigmoid(scores)

        # Update weights with log likelihood gradient
        errors = classIDs - predictions

        gradient = np.dot(features.T, errors)
        params += learning_rate * gradient

        if np.linalg.norm(gradient) < 0.1:
            #print("Gradient is close to 0, reached convergence.")
            #print("Final Loglikelihood:")
            #print(log_likelihood(features, classIDs, params))
            break

		# Print log-likelihoods
		#if step % 10000 == 0:
		#	print(log_likelihood(features, classIDs, params))

    return params

def test(params, data = 'classify_d5_k3_saved1.mat'):
    mat = scipy.io.loadmat(data)
    test_features, test_classIDs = format_data(mat, train = False)
    test_scores = np.dot(test_features, params)
    preds = sigmoid(test_scores)
    acc = sum(np.round(preds) == test_classIDs) / len(preds)
    return acc

def test_sklearn(data = 'classify_d5_k3_saved1.mat'):
    mat = scipy.io.loadmat(data)    
    features, classIDs = format_data(mat)
    test_features, test_classIDs = format_data(mat, train = False)
    clf = LogisticRegression(fit_intercept=True, C = 1e15)
    clf.fit(features[:,1:], classIDs)
    acc_sklearn = clf.score(test_features[:,1:], test_classIDs)
    return(acc_sklearn)


