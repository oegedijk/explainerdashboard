"""
For the @properties in our ClassifierBunch class we would like users to be able to call them as a property
and get the property for the pos_label set for the entire class (e.g. if 1 is the positive class, then
explainer.shap_values will return the shap_values for class 1). However it should also be possible
to specify a specific pos_label by making the @property callable: explainer.shap_values(0) will return the shap 
values for class 0.

For this we introduce two new classes: 
1. DefaultCOlumns2dNpArray, which when called as a property returns a specific default column of a 
2d np.ndarray, but when called as a callable can return any column.
3. DefaultList, which when called as a property returns a specific default item of a list, 
but when called as a callable can return any item.

To maintain compatibility in interfaces between ClassifierBunch and RegressionBunch, we make all the properties
callable using make_callable, which adds a dummy __call__ method to the underlying object. 

"""
from typing import List

import numpy as np
import pandas as pd


class CallablePdSeries(pd.Series):
    """a pd.Series object with a dummy .__call__() method"""
    def __call__(self, *args, **kwargs):
        return pd.Series(self)
    
class CallablePdDataFrame(pd.DataFrame):
    """a pd.DataFrame object with a dummy .__call__() method"""
    def __call__(self, *args, **kwargs):
        return pd.DataFrame(self)
    
class CallableNpArray(np.ndarray):
    """a np.ndarray object with a dummy .__call__() method"""
    def __new__(cls, arr):
        obj = np.asarray(arr).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        
    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context).view(np.ndarray)

    def __call__(self, *args, **kwargs):
        return self.view(np.ndarray)
    
class CallableList(list):
    """a list object with a dummy .__call__() method"""
    def __call__(self, col=None):
        return list(self)


def make_callable(obj):
    """
    returns the obj with a dummy __call__ method attached.
    This means that calling `obj` and 'obj(()' and  'obj(arg1, arg2='foo') result in the same thing.
    
    Used to make @property that in subclass may be callable.
    """
    if isinstance(obj, np.ndarray):
        return CallableNpArray(obj)
    if isinstance(obj, pd.Series):
        return CallablePdSeries(obj)
    if isinstance(obj, pd.DataFrame):
        return CallablePdDataFrame(obj)
    if isinstance(obj, list):
        return CallableList(obj)
    
    class DummyCallable(type(obj)):
        def __call__(self, *args, **kwargs):
            return type(obj)(self)
    
    return DummyCallable(obj)

class DefaultDfList(pd.DataFrame):
    """"
    You have the set source_list manually!

    e.g. 

    dfl = DefaultDfList(df1)
    dfl.source_list = [df1, df2]
    """
    _internal_names = list(pd.DataFrame._internal_names) + ['source_list']
    _internal_names_set = set(_internal_names)

    def __call__(self, index=None):
        if index is not None:
            return self.source_list[index]
        else:
            return pd.DataFrame(self)

    @property
    def _constructor(self):
        return DefaultDfList


class DefaultSeriesList(pd.Series):
    _internal_names = list(pd.Series._internal_names) + ['source_list']
    _internal_names_set = set(_internal_names)

    def __call__(self, index=None):
        if index is not None:
            return self.source_list[index]
        else:
            return pd.Series(self)

    @property
    def _constructor(self):
        return DefaultSeriesList


class DefaultNpArrayList(np.ndarray):
    def __new__(cls, default_array, source_list):
        obj = np.asarray(default_array).view(cls)
        obj.source_list = source_list
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.source_list = getattr(obj, 'source_list', None)
        
    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context).view(np.ndarray)

    def __call__(self, index=None):
        if index is not None:
                return self.source_list[index]
        return self.view(np.ndarray)


def default_list(source_list:List, default_index:int):
    """
    Normally gives the default_index item in a list.
    If used as a callable, you can specify a specific index.
    
    Use to make @property that you can pass optional index parameter to
    """

    if isinstance(source_list[default_index], pd.DataFrame):
        df_list = DefaultDfList(source_list[default_index])
        df_list.source_list = source_list
        return df_list

    if isinstance(source_list[default_index], pd.Series):
        s_list = DefaultSeriesList(source_list[default_index])
        s_list.source_list = source_list
        return s_list

    if isinstance(source_list[default_index], np.ndarray):
        a_list = DefaultNpArrayList(source_list[default_index], source_list)
        return a_list

    class DefaultList(type(source_list[default_index])):
        def __new__(cls, default_value, source_list):
            obj = type(source_list[default_index]).__new__(cls, default_value)
            return obj

        def __init__(self, default_value, source_list):
            super().__init__()
            self.source_list = source_list
            self.default_type = type(default_value)

        def __call__(self, index=None):
            if index is not None:
                return self.source_list[index]
            else:
                
                return self.default_type(self)
    
    return DefaultList(source_list[default_index], source_list)


class DefaultColumn2dNpArray(np.ndarray):
    """
    a 2d numpy array that by default returns only a single default column array.
    If used as a callable, you can specify a specific column to return. 
    
    Used to make @property that you can pass a column parameter to
    """
    def __new__(cls, full_array, default_col):
        obj = np.asarray(full_array[:, default_col]).view(cls)
        obj.full_array = full_array
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.full_array = getattr(obj, 'full_array', None)
        
    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context).view(np.ndarray)

    def __call__(self, col=None):
        if col is not None:
            return self.full_array[:, col]
        return self.view(np.ndarray)


def default_2darray(array_2d:np.ndarray, default_column:int):
    """
    when used as property returns default_column col of array_2d
    when used as callable __call__(col) returns column col of array_2d
    
    Used to make a @property that you can pass an optional column parameter to
    """
    return DefaultColumn2dNpArray(array_2d, default_column)