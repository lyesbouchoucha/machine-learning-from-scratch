import numpy as np

class MyKNNRegressor:
    def __init__(self, n_neighbors=5):
        self.k = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X_new):
      
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fitted before calling predict.")
            
        X_new = np.array(X_new)
        predictions = []
        
        for x in X_new:
       
            distances = np.linalg.norm(self.X_train - x, axis=1)
            
            k_indices = np.argsort(distances)[:self.k]
            
            k_nearest_labels = self.y_train[k_indices]
          
            predictions.append(np.mean(k_nearest_labels))
            
        return np.array(predictions)

