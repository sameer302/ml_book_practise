import matplotlib.pyplot as plt # plotting library
import numpy as np
from load_iris_dataset import df
# to make a folder into a module we need to have __init__.py file in that folder.

# select setosa and versicolor
y = df.iloc[0:100, 4].values
''' 
Selects rows 0-99
Takes column index 4 (class label)
Converts to NumPy array.
'''
y = np.where(y == 'Iris-setosa', 0, 1)
'''
Converts labels to numbers:
Iris-setosa → 0
Others (versicolor) → 1
'''

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
'''
Selects sepal length (col 0)
Selects petal length (col 2)
For first 100 samples
Converts to NumPy array.
'''
if __name__ == '__main__':
    # plot data
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    '''
    Plots first 50 points (Setosa):
    X-axis → sepal length
    Y-axis → petal length
    Red circles.
    '''
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='versicolor')
    '''
    Plots next 50 points (Versicolor):
    Blue squares.
    '''
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')

    plt.savefig('iris_dataset.png')
    plt.show()

'''
above we added if __name__ == '__main__': block so that when this script is imported as a module in other scripts,
the code inside this block will not be executed.
Why this works
Imported file → __name__ = training_data_prep
Direct run → __name__ = "__main__"
So plot runs only when you run the file directly.
'''