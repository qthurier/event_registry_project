from sklearn.base import TransformerMixin 
from sklearn.preprocessing import FunctionTransformer, add_dummy_feature

class ColumnExtractor(TransformerMixin):
    """Extract columns from a pandas dataframe in a pipeline """
    def __init__(self, col, *args, **kwargs):
        self.col = col
    def transform(self, X, y=None):
        return X[self.col].fillna('missing')
    def fit(self, X, y=None, *args, **kwargs):
        return self

import numpy as np
import collections, itertools

class MostCommonEntity(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.most_commons = None
    def fit(self, X, y=None):
        occurence = collections.Counter(list(itertools.chain(*X)))
        self.most_commons = [k for k, v in occurence.most_common()[:500]]
        return self
    def transform(self, X, y=None):
        val = []
        for x in X:
            val.append([int(e in x) for e in self.most_commons])
        return np.array(val)