import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal

class GaussNB:
    
    def __init__(self):
        pass
    
    def _mean(self,X): 
        mu = dict()
        for i in self.classes_:
            idx = np.argwhere(self.y == i).flatten()
            mean = []
            for j in range(self.n_feats):
                mean.append(np.mean( X[idx,j] ))
            mu[i] = mean
        return mu
        
    def _cov(self,X):
        
        cov = dict()
        for i in self.classes_:
            idx = np.argwhere(self.y==i).flatten()
            cov[i] = np.cov(X[idx,:].T)
        return cov
            
    
    def _prior(self):
        P = {}
        for i in self.classes_:
            P[i] = 0.5
        return P
    
        
    def P_E_H(self,x,h):
        
        return multivariate_normal.pdf(x, mean= self.means_[h], cov=self.cov_[h])
        
        
    def fit(self, X, y):
        self.n_samples, self.n_feats = X.shape
        self.n_classes = np.unique(y).shape[0]
        self.classes_ = np.unique(y)
        self.y = y
        
        self.means_ = self._mean(X) # dict of list {class:feats}
        self.priors_ = self._prior()
        self.cov_ = self._cov(X)
        
    def predict(self,X):
        samples, feats = X.shape
        if feats!=self.n_feats:
            print("No dimension match with training data!")
            
        result = []
        for i in range(samples):
            distinct_likelyhoods = []
            for h in self.classes_:
                tmp = self.P_E_H(X[i],h)
                distinct_likelyhoods.append( tmp * self.priors_[h])
            marginal = np.sum(distinct_likelyhoods)
            tmp = 0
            probas = []
            for h in self.classes_:
                numerator =  distinct_likelyhoods[tmp]
                denominator = marginal
                probas.append( numerator / denominator )
                tmp+=1
            # predicting maximum
            idx = np.argmax(probas)
            result.append(self.classes_[idx])
        return result