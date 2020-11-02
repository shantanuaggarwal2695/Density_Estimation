import preprocess as prop
import pca
import multivariate_gaussian as mvg
import sys
import scipy

data_dir = "./data/"

def executeProj():
    TrainClass0, TrainClass1, TestClass0, TestClass1 = sys.argv
    class0_train = scipy.io.loadmat(data_dir + "training0.mat") 
    class1_train = scipy.io.loadmat(data_dir + "training1.mat")
    class0_test = scipy.io.loadmat(data_dir + "testing0.mat") 
    class1_test = scipy.io.loadmat(data_dir + "testing1.mat")

    model = train(class0_train, class1_train)
    predicted = test(class0_test, class1_test)
    print("Accuracy:"+ accuracy())

def train(class0, class1):
    


def test():


def accuracy():









if __name__ == '__main__':
    executeProj()