import numpy as np
class Perceptron:
    '''
Perceptron classifier.

Parameters
------------
eta : float
Learning rate (between 0.0 and 1.0)

n_iter : int
Passes over the training dataset.

random_state : int
Random number generator seed for random weight initialization.

Attributes
-----------
w_ : 1d-array
Weights after fitting.

b_ : Scalar
Bias unit after fitting.

errors_ : list
Number of misclassifications (updates) in each epoch.

    '''

def __init__(self, eta=0.01, n_iter=50, random_state=1):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state

def fit(self, X, y):
    """
    Fir training data.

    Parameters
    ----------
    X : {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape = [n_samples]
        Target values.

    Returns
    -------
    self : object
    """

    '''
    below we create a random number generator object/instance.
    the seed value is provided by the user while creating the perceptron object.
    using same seed value each time ensures that same random numbers are generated every time.
    rgen will act like an independent random number generator.
    further we can choose the random number from any distribution using this rgen object.
    for e.g., rgen.normal(), rgen.uniform() etc.
    '''
    rgen = np.random.RandomState(self.random_state)
    '''
    Now while using rgen to initialize weights, we are using normal distribution with mean 0.0 and standard deviation 0.01. We can also use other distributions like uniform distribution.
    size parameter is used to specify the shape of the output array.
    Here we are initializing weights for all features, hence size=X.shape[1] (number of features).
    If we had written size=X.shape[0], then weights would have been initialized for number of samples instead of features.
    '''
    self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
    self.b_ = 0.0
    self.errors_ = []

    for _ in range(self.n_iter):
        errors = 0
        '''
        here zip pairs elements index-wise from X and y.
        so it creates tuples: (X[0], y[0]), (X[1], y[1]), .....
        xi is one training sample and has shape (n_features, ) e.g., [1,2]
        target is the corresponding label for that training sample e.g., 1 or 0
        so the below loop means, loop over each data point with its label at the same time 
        '''
        for xi, target in zip(X, y):
            update = self.eta * (target - self.predict(xi))
            self.w_ += update * xi
            self.b_ += update
            errors += int(update != 0.0)
        self.errors_.append(errors)
    return self

def net_input(self, X):
    """Calculate net input"""
    return np.dot(X, self.w_) + self.b_

def predict(self, X):
    """Return class label after unit step"""
    return np.where(self.net_input(X) >= 0.0, 1, 0)