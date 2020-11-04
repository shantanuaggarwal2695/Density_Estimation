import preprocess as prop
import pca
import multivariate_gaussian as mvg
import sys
from scipy import io

data_dir = "./data/"

class execute:
    def __init__(self):
        self.dim_red  = pca.PCA()


    def norm_train(self, class0, class1):
        p = prop.Process(class0, class1)
        self.train = p.preprocess()
        self.mu = self.train.mean()
        self.sigma = self.train.std()
        self.norm, self.label = p.normalize(flag=True)
        return self.norm, self.label
    
    def norm_test(self, class0, class1):
        p = prop.Process(class0, class1)
        self.test = p.preprocess()
        self.norm, self.label = p.normalize(self.mu, self.sigma, flag=False)
        return self.norm, self.label
    
    def fit_pca(self):
        final_train = self.dim_red.fit_transform(self.norm)

    def transform_pca(self):
        final_test = self.dim_red.transform(self.norm)

class train:

    def __init__(self, train, labels):
        self.train = train
        self.labels = labels
    
    def estimate(self):
        self.gnb = mvg.GaussNB()
        self.gnb.fit(self.train, self.labels)

    def predict(self, test, labels):
        predicted = self.gnb.predict(test)
        


if __name__ == '__main__':

    TrainClass0, TrainClass1, TestClass0, TestClass1 = sys.argv[1:5]
    class0_train = io.loadmat(data_dir + TrainClass0) 
    class1_train = io.loadmat(data_dir + TrainClass1)
    class0_test = io.loadmat(data_dir + TestClass0) 
    class1_test = io.loadmat(data_dir + TestClass1)

    # print(class0_train)
    #Start processing of Data
    exec = execute()
    train,label = exec.norm_train(class0_train, class1_test)
    # print(train)


    