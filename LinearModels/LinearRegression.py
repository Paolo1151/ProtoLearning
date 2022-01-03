import numpy as np

class LinearRegression:
    """
    A class that represents a Linear Regression Model

    Attributes
    ----------
    iterations : int
        number of training iterations
    learning_rate : float
        alpha multiplier for the strength of the gradient descent
    seed : int, list, etc. : Default - None
        random seed for reproducibility

    Methods
    -------
    train(X, y):
        Trains the model based on the data and labels
 
    predict(X):
        Predict the Labels for each provided X input data
    """
    def __init__(self, iterations=1000, learning_rate=0.01, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        
        self.iterations = iterations
        self.learning_rate = learning_rate
        
    def _calculate_gradient(self, X, y):
        """
        Calculate the Gradient Vector for gradient descent
        """
        grad = []
        for i in range(X.shape[1]):
            grad.append(self._calculate_cost_der(X, y, i, X.shape[1]))
        return np.array(grad)

    def _calculate_cost(self, X, y, m):
        """
        Calculate the MSE / Cost Function based on the current theta values
        """
        result = (X@self.theta - y)@(X@self.theta - y)
        result *= (1/(2*m))
        return result

    def _calculate_cost_der(self, X, y, j, m):
        """
        Calculate the Partial Derivative of the Cost Function
        """
        result = (X@self.theta - y) @ X[:, j]
        result *= (1/m)
        return result

    def train(self, X, y):
        """
        Trains the model to learn the parameters
        """
        self.theta = np.random.rand(X.shape[1])

        for i in range(self.iterations):
            grad = self._calculate_gradient(X, y)
            self.theta -= self.learning_rate * grad
        
    def predict(self, X):
        """
        Predict the Labels for each provided X input data
        """
        preds = []
        for d in X:
            preds.append((self.theta @ d)[0])
        return np.array(preds)

