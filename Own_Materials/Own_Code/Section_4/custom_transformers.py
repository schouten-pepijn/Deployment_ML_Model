import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

"""
    add pydantic type checkings
"""

class TemporalVariableTransformer(
    BaseEstimator, TransformerMixin
):
    def __init__(
        self,
        variables: list,
        reference_variables
    ):
        if not isinstance(variables, list):
            raise ValueError("Variables should be a list")
        
        self.variables = variables
        self.reference_variables = reference_variables
        
    def fit(
        self,
        X,
        y=None
    ):
        return self
    
    def transform(
        self,
        X
    ):
        
        
