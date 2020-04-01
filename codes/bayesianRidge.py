import pandas as pd
import numpy as np

from scipy.stats import norm
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split

### Basic linear regression using the parameters to estimate the current impulse

class BR():

    def __init__(self, cellId, dfs, random_state=15):
        
        self.cellId = cellId
        self.dfs = dfs
        self.id_df = None
        if dfs is not None: self.labels = dfs[0].columns

    def _fit(self, X, y):
        # Train a linear regression model
        lr = BayesianRidge()
        lr.fit(X, y)

        return lr

    def _predict(self, id_, df):
        # Predict values according to id_ model and inputs params

        def generate_inputs(df):
            # Transfer df to current X
            if len(df.shape) == 1:
                return df.values[self.params_cols], df['threshold']
            return df.iloc[:, self.params_cols].values, df['threshold'].values

        X, thrs = generate_inputs(df)

        # If inputs is 1-d, transfer to 2-d
        if len(X.shape) == 1: X = X.reshape(1, -1)

        means, stds = self.id_df[id_].predict(X, return_std=True)

        # the first column for class 0, the second column for class 1
        probs = np.zeros((len(means),2))

        # Transfer to probability distribution
        for i in range(len(means)):
            mean, std = means[i], stds[i]
            th = thrs[i]
            probs[i,0] = norm(mean,std).cdf(th)
            probs[i,1] = 1 - probs[i,0]

        return probs

    def choose_cols(self, params, target):
        # Decide which columns to use as params
        # and which column as target
        self.params_cols = []

        for i, label in enumerate(self.labels):
            if label in params:
                self.params_cols.append(i)
            if label == target:
                self.target_col = i

    def fit(self, params, target):
        # Choose the params and train the linear model for each grid
        # Target is a pandas dataframe

        # Get the column number of params and target
        self.choose_cols(params, target)

        self.id_df = dict()
        for id_, df in zip(self.cellId, self.dfs):
            # acquire X and y, and do linear regression
            X, y = df.iloc[:,self.params_cols], df.iloc[:,self.target_col]
            lr = self._fit(X, y)
            # Store 
            self.id_df[id_] = lr
        
    def predict(self, cellIds, dfs):
        # Choose the cellIds and inputs to do predict
        # Output results with shape (ids_nums, data_nums)

        if len(cellIds) != len(dfs):
            print('The number of cells and inputs must be consistant')
            return None
        
        results = []
        for id_, df in zip(cellIds, dfs):
            # Do prediction for each cellid
            results.append(self._predict(id_, df))

        return results





