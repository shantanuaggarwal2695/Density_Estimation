import numpy as np
import numpy.linalg as la

class PCA:

    def fit_transform(self, df):
        cov_mat = np.array(df.cov())
        evals, evects = la.eig(cov_mat)
        eig_pairs = [(np.abs(evals[i]), evects[:,i]) for i in range(len(evals))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        self.eigen_vectors = np.hstack((eig_pairs[0][1].reshape(784,1), 
                      eig_pairs[1][1].reshape(784,1)
                          ))
        train_components = np.array(df).dot(self.eigen_vectors)
        return train_components
    
    def transform(self, df):
        test_components = np.array(df).dot(self.eigen_vectors)
        return test_components









