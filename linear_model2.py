from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# define a function
# takes in X and y and returns the best theta value
# according to our "direct solution" method
def optimal_theta(X,y):
    XTX = np.dot(X.T,X)
    XTX_inv = np.linalg.inv(XTX)
    XTy = np.dot(X.T,y)

    return np.dot(XTX_inv,XTy)

# define our loss function, in terms of theta, theta wil be 11x10, X will be 420x11, y is about 400x1, pred will also be about 400x1
def loss(theta, X, y):
    # get the predictions
    pred = predict(theta,X)

    # get the difference with the true values
    diffs = y -  pred

    # could return np.dot(diffs,T,diffs)

    # square all the diffs:
    sqdiffs = np.power(diffs,2.0)

    # return the mean squared error
    return sqdiffs.mean()

# predict given X and theta
def predict(theta, X):
    return np.dot(X,theta)

# preprocessing of X
def preproc(X):
    # subtracting off the mean
    X = X - X.mean(0)
    # normalizing the dataset
    X = X / X.std(0)
    # run PCA on the data
    X = PCA(X.shape[1]).fit_transform(X)
    
    # shape = 3x + 1
    # feature extraction
    return np.concatenate(
        [np.power(X,2.0), np.cos(X), X,np.ones((len(X),1))],
        1
    )

# preprocessing of X with variable number of components for PCA
def preproc_with_comp(X,comp):
    # subtracting off the mean
    X = X - X.mean(0)
    # normalizing the dataset
    X = X / X.std(0)
    # run PCA on the data
    # X = PCA(X.shape[1]).fit_transform(X)
    X = PCA(comp).fit_transform(X)
    
    # shape = 3x + 1
    # feature extraction
    return np.concatenate(
        [np.power(X,2.0), np.cos(X), X,np.ones((len(X),1))],
        1
    )

# preprocessing of X
def preproc_with_feat(X,feat):
    # subtracting off the mean
    X = X - X.mean(0)
    # normalizing the dataset
    X = X / X.std(0)
    # run PCA on the data
    X = PCA(X.shape[1]).fit_transform(X)
    
    # shape = 3x + 1
    # feature extraction
    if feat == 1:
        return np.concatenate(
            [np.power(X,2.0)],
            1
        )
    elif feat == 2:
        return np.concatenate(
            [np.power(X,2.0),
             np.cos(X)],
            1
        )
    elif feat == 3:
        return np.concatenate(
            [np.power(X,2.0),
             np.cos(X),
             X],
            1
        )
    else:
        return np.concatenate(
            [np.power(X,2.0),
             np.cos(X),
             X,
             np.ones((len(X),1))],
            1
        )


print("Diabetes Dataset")
# import our dataset
X,y = load_diabetes(return_X_y= True, scaled=False)

print("X.shape,y.shape:",X.shape,y.shape)
Xproc = preproc(X)
print("Xproc:",Xproc)
print("Xproc.shape:",Xproc.shape)

# # plt.scatter(X[:,2],y)
# plt.scatter(Xproc[:,2],y)
# plt.show()

# random_theta = np.random.random((11,1))
random_theta = np.random.random((31,1))
best_theta = optimal_theta(Xproc,y)

print("loss(random_theta,Xproc,y):",loss(random_theta,Xproc,y))
print("loss(best_theta,Xproc,y):",loss(best_theta,Xproc,y))

print("")
print("Variable PCA Components Experiments:")
for i in range(1,X.shape[1]+1):
    Xproc = preproc_with_comp(X,i)
    # random_theta = np.random.random((11,1))
    best_theta = optimal_theta(Xproc,y)
    print("loss(best_theta,Xproc,y) with",i,"components:",loss(best_theta,Xproc,y))

print("")
print("Variable Number of Features Extracted Experiments:")
for i in range(1,5):
    Xproc = preproc_with_feat(X,i)
    best_theta = optimal_theta(Xproc,y)
    print("loss(best_theta,Xproc,y) with",i,"features extracted:",loss(best_theta,Xproc,y))

print("")
print("---------------------------------------------------------------------------")

print("Breast Cancer Dataset")
# import our dataset
X,y = load_breast_cancer(return_X_y= True)

print("X.shape,y.shape:",X.shape,y.shape)
Xproc = preproc(X)
print("Xproc:",Xproc)
print("Xproc.shape:",Xproc.shape)

# random_theta = np.random.random((11,1))
random_theta = np.random.random((91,1))
best_theta = optimal_theta(Xproc,y)

print("loss(random_theta,Xproc,y):",loss(random_theta,Xproc,y))
print("loss(best_theta,Xproc,y):",loss(best_theta,Xproc,y))

print("")
print("Variable PCA Components Experiments:")
for i in range(1,X.shape[1]+1):
    Xproc = preproc_with_comp(X,i)
    # random_theta = np.random.random((11,1))
    best_theta = optimal_theta(Xproc,y)
    print("loss(best_theta,Xproc,y) with",i,"components:",loss(best_theta,Xproc,y))

print("")
print("Variable Number of Features Extracted Experiments:")
for i in range(1,5):
    Xproc = preproc_with_feat(X,i)
    best_theta = optimal_theta(Xproc,y)
    print("loss(best_theta,Xproc,y) with",i,"features extracted:",loss(best_theta,Xproc,y))


