import matplotlib.pyplot as plt
import numpy as np
from training_data_prep import X, y
from train_perceptron import ppn

def plot_decision_regions(X, y, classifier):
        plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
        plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='versicolor')
        
        # Create smooth x-values for decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x_line = np.linspace(x_min, x_max, 100)
        # Decision boundary: w_[0]*x1 + w_[1]*x2 + b = 0, solve for x2
        y_line = -(classifier.w_[0] * x_line + classifier.b_) / classifier.w_[1]
        plt.plot(x_line, y_line, color='black', linestyle='--', label='Decision Boundary')
                
        plt.xlabel('sepal length [cm]')
        plt.ylabel('petal length [cm]')
        plt.legend(loc='upper left')
        plt.title('Perceptron Decision Boundary')
        plt.savefig('decision_boundary.png')
        plt.show()
if __name__ == '__main__':
    plot_decision_regions(X, y, classifier=ppn)