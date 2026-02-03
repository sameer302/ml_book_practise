import matplotlib.pyplot as plt # plotting library
import numpy as np
from load_iris_dataset import df
'''
While using imports, its better to have __init__.py file in the folder from which we are importing. This treats that folder
as a module or package. Further, if we have subfolders inside that folder, then those subfolders should also have __init__.py file.
This ensures that all the subfolders are also treated as sub-modules or sub-packages and we can import from them without any issues.

Further there are two types of imports:
1. Absolute imports: In this we provide the complete path of the module from the root folder.
   e.g., from chapter1.Iris_dataset_perceptron.load_iris_dataset import df
2. Relative imports: In this we provide the path of the module relative to the current module.
   e.g., from .load_iris_dataset import df
Here we are using relative import.

Also whenever we use relative or absolute imports, a file named __pycache__ is created in the folder from which we are importing.
This file contains the compiled bytecode of the imported module. This helps in faster loading of the module in subsequent imports.
'''

# to convert a folder into a module we need to have __init__.py file in that folder, but if we are importing from the same

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