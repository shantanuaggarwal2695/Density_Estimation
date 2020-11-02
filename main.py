import preprocess as prop
import pca
import multivariate_gaussian as mvg
import sys
import scipy

data_dir = "./data/"

class execute:
    def __init__(self):
        self.dim_red  = pca.PCA()


    def norm_train(self, class0, class1):
        p = prop.Process(class0, class1)
        self.train,self.label = p.preprocess()
        self.mu = self.train.mean()
        self.sigma = self.train.std()
        self.norm = p.normalize(flag=True)
    
    def norm_test(self, class0, class1):
        p = prop.Process(class0, class1)
        

        
        

    
    def fit_pca(self):
        final_train = self.dim_red.fit_transform(self.norm)

    def transform_pca(self):
        final_test = self.dim_red.transform()



        

















if __name__ == '__main__':
    TrainClass0, TrainClass1, TestClass0, TestClass1 = sys.argv
    class0_train = scipy.io.loadmat(data_dir + "training0.mat") 
    class1_train = scipy.io.loadmat(data_dir + "training1.mat")
    class0_test = scipy.io.loadmat(data_dir + "testing0.mat") 
    class1_test = scipy.io.loadmat(data_dir + "testing1.mat")

    