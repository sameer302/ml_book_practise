import pandas as pd

s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(s, header=None, encoding='utf-8')
'''
Here header=None indicates that the CSV file does not contain a header row or column names row
,so pandas will assign default integer column names (0, 1, 2, ...) and will not assume first row as column names.

Further, encoding='utf-8' specifies that the file is encoded in UTF-8 format, which is a common character encoding standard.
This ensures that pandas correctly interprets the characters in the file, especially if it contains special or non-ASCII characters.

Here we have loaded the datatset from URL but if we download the csv file then we can read it using the below code:
df = pd.read_csv(file_path, header=None, encoding='utf-8')
'''
if __name__ == '__main__':
    print(df.shape)
    print(df.tail())