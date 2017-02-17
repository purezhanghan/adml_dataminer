import pandas as pd

# Load data file


def data_loader(path):
    file = pd.read_csv(path)
    return file

if __name__ == "__main__":
    print('Loading data.')
    train = data_loader('train.csv')
    test = data_loader('test.csv')
    data = train.append(test)
    print('Finish loading.')


