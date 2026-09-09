import numpy as np

class MyLinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        if self.fit_intercept:
            X_design = np.hstack((np.ones((X.shape[0], 1)), X))
        else:
            X_design = X

        A = X_design.T @ X_design
        b = X_design.T @ y
        
        beta = np.linalg.solve(A, b)

        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model must be fitted before calling predict.")
  
        return X @ self.coef_ + self.intercept_
