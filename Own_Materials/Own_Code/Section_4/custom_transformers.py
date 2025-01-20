import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from typing import List, Dict

"""
    add pydantic type checkings
"""

class TemporalVariableTransformer(
    BaseEstimator,
    TransformerMixin
):
    # Temporal elapsed time transformer (scikit-learn)
    def __init__(
        self,
        variables: List,
        reference_variable
    ):
        # sanity check
        if not isinstance(variables, list):
            raise ValueError("Variables should be a list")
        
        self.variables = variables
        self.reference_variable = reference_variable
        
    def fit(
        self,
        X,
        y=None
    ):
        # dummy fit for scikit-learn
        return self
    
    def transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        # not to overwrite the original df
        X = X.copy()
        
        # loop over all the features
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]



class StringMapper(
    BaseEstimator,
    TransformerMixin
):
    # categorical missing value imputer
    def __init__(
        self,
        variables: List,
        mappings: Dict
    ):
        
        # sanity check
        if not isinstance(variables, list):
            raise ValueError("Variables should be a list")
        
        self.variables = variables
        self.mappings = mappings
    
    def fit(
        self,
        X,
        y=None
    ):
        # dummy fit for scikit-learn
        return self
    
    def transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        
        # not to overwrite the original df
        X = X.copy()
        
        # apply the mapping on each variable
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings)
            
        return X


class MeanImputer(
    BaseEstimator,
    TransformerMixin
):
    
    def __init__(
        self,
        variables
    ):
        
        # sanity check
        if not isinstance(variables, list):
            raise ValueError("Variables should be a list")
        
        self.variables = variables
        
    def fit(
        self,
        X,
        y=None
    ):
        # persist mean values in a dictionary
        self.imputer_dict_ = (
            X[self.variables]
            .mean()
            .to_dict()
        )
        
        return self
    
    
    def transform(
        self,
        X
    ):

        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna(self.imputer_dict_[feature])
            
        return X
    
    
    
    class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
        # Groups infrequent categories into a single string

        def __init__(self, tol=0.05, variables=None):

            if not isinstance(variables, list):
                raise ValueError('variables should be a list')
            
            self.tol = tol
            self.variables = variables

        def fit(self, X, y=None):
            # persist frequent labels in dictionary
            self.encoder_dict_ = {}

            for var in self.variables:
                # the encoder will learn the most frequent categories
                t = pd.Series(X[var].value_counts(normalize=True))
                # frequent labels:
                self.encoder_dict_[var] = list(t[t >= self.tol].index)

            return self

        def transform(self, X):
            X = X.copy()
            for feature in self.variables:
                X[feature] = np.where(
                    X[feature].isin(
                        self.encoder_dict_[feature]
                    ),
                    X[feature],
                    "Rare"
                )

            return X
        
        
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    # String to numbers categorical encoder

    def __init__(self, variables):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        
        self.variables = variables

    def fit(self, X, y):
        temp = pd.concat(
            [X, y], axis=1
        )
        temp.columns = list(X.columns) + ["target"]

        # persist transforming dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            t = (temp
                 .groupby([var])["target"]
                 .mean()
                 .sort_values(ascending=True)
                 .index
            )
            self.encoder_dict_[var] = {
                k: i for i, k in enumerate(t, 0)
            }

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        return X