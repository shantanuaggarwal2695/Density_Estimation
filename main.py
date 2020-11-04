import preprocess as prop
import pca
import multivariate_gaussian as mvg
import sys
from scipy import io
import numpy as np
from sklearn.metrics import accuracy_score

data_dir = "./data/"

class execute:
    def __init__(self):
        self.dim_red  = pca.PCA()


    def norm_train(self, class0, class1):
        p = prop.Process(class0, class1)
        self.train = p.preprocess()
        self.mu = self.train.mean()
        self.sigma = self.train.std()
        self.norm_train, self.label_train = p.normalize(flag=True)
        return self.norm_train, self.label_train
    
    def norm_test(self, class0, class1):
        p = prop.Process(class0, class1)
        self.test = p.preprocess()
        self.norm_test, self.label_test = p.normalize(self.mu, self.sigma, flag=False)
        return self.norm_test, self.label_test
    
    def fit_pca(self):
        final_train = self.dim_red.fit_transform(self.norm_train)
        return final_train

    def transform_pca(self):
        final_test = self.dim_red.transform(self.norm_test)
        return final_test

class estimator:

    def __init__(self, train, labels):
        self.train = train
        self.labels = labels
    
    def estimate(self):
        self.gnb = mvg.GaussNB()
        self.gnb.fit(self.train, self.labels)

    def predict(self,test):
        predicted = self.gnb.predict(test)
        return predicted

    def accuracy(self, truth, predicted):
        print(accuracy_score(truth, predicted))
        


if __name__ == '__main__':

    TrainClass0, TrainClass1, TestClass0, TestClass1 = sys.argv[1:5]
    class0_train = io.loadmat(data_dir + TrainClass0) 
    class1_train = io.loadmat(data_dir + TrainClass1)
    class0_test = io.loadmat(data_dir + TestClass0) 
    class1_test = io.loadmat(data_dir + TestClass1)

    #Start processing of Data
    exec = execute()
    train,label1 = exec.norm_train(class0_train, class1_train)
    test,label2 = exec.norm_test(class0_test, class1_test)

    # print("Processing Done.......")
    train_red = exec.fit_pca()
    test_red = exec.transform_pca()

    # print("PCA done........")

    training = estimator(train_red, np.array(label1))
    training.estimate()
    # print("Model trained..........")
    predicted = training.predict(test=test_red)
    training.accuracy(label2, predicted)

    





    