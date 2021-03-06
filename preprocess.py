import numpy as np
import pandas as pd


class Process:

    def __init__(self, class_0, class_1):
        self.class0 = class_0['nim0']
        self.class1 = class_1['nim1']
    
    def preprocess(self):
        training_0 = np.reshape(self.class0, (784,-1))
        training_1 = np.reshape(self.class1, (784,-1))
        training_0 = np.transpose(training_0)
        training_1 = np.transpose(training_1)
        df0 = pd.DataFrame(training_0)
        df1 = pd.DataFrame(training_1)
        df0_label = [0]*df0.shape[0]
        df1_label = [1]*df1.shape[0]

        self.df_train = pd.concat([df0,df1], axis=0)
        self.df_label = df0_label + df1_label
        return self.df_train
    
    def normalize(self, *args, flag=True):
        if flag:
            df_norm = (self.df_train - self.df_train.mean())/self.df_train.std()
        else:
            df_norm = (self.df_train - args[0])/args[1]
        
        return df_norm, self.df_label

















        





