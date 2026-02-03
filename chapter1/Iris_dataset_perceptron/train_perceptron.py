from perceptron import Perceptron
import matplotlib.pyplot as plt
from training_data_prep import X, y
# when we import any module, the code present in that module and the modules imported in that module is executed.

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

if __name__ == '__main__':
    # plot number of misclassifications in each epoch
    plt.plot(range(1, (ppn.n_iter) + 1), ppn.errors_, marker='o')
    '''
    above
    plt.plot makes a line plot while plt.scatter makes a scatter plot.
    '''
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates') 
    plt.show()