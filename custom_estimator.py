from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class TfIdfCustomized(BaseEstimator):
    def __init__(self, estimator=TfidfVectorizer()):
        self.estimator=estimator
    def fit(self, X, y=None, **kwargs):
        return self.estimator.fit(X,y)
    def predict(self, X, y=None):
        return self.estimator.predict(X)
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    def transform(self, X):
        X_ = self.estimator.transform(X)
        return pd.DataFrame(X_.todense(), columns=self.estimator.get_feature_names())
    def fit_transform(self, X, y=None):
        X_ = self.estimator.fit_transform(X)
        return pd.DataFrame(X_.todense(), columns=self.estimator.get_feature_names())
    


