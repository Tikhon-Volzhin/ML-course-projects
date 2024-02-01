from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin 
from sklearn.base import RegressorMixin
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from typing import Optional, List
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np

class BaseDataPreprocessor(TransformerMixin):
    def __init__(self, needed_columns=None):
        super().__init__()
        self.scaler = StandardScaler()
        self.needed_columns = needed_columns

    def fit(self, data, *args):
        if self.needed_columns is None:
            self.needed_columns = list(data.columns)

        transformed_data = data[self.needed_columns]
        transformed_data = self.scaler.fit_transform(X = transformed_data)
        return self
        

    def transform(self, data: pd.DataFrame) -> np.array:
        transformed_data = data[self.needed_columns]
        transformed_data = self.scaler.transform(transformed_data)
        return np.array(transformed_data)
    

def root_mean_squared_logarithmic_error(y_true, y_pred, a_min=1.):
    np_y_true, np_y_pred = np.array(y_true), np.array(y_pred)
    np_y_pred = np.where(np_y_pred > a_min, np_y_pred, a_min)
    result = (np.log(np_y_true) - np.log(np_y_pred))**2
    result = np.sqrt(result.mean())
    return result


class ExponentialLinearRegression(Ridge):
    def __init__(self, **kwargs):
        super().__init__( **kwargs)

    def fit(self, X, Y):
        log_Y_true = np.log(Y)
        super().fit(X, log_Y_true)

    def predict(self, X):
        log_Y_pred = super().predict(X)
        return np.exp(log_Y_pred)
    

class SGDLinearRegressor(RegressorMixin):
    def __init__(self, 
                lr=0.01, 
                regularization=1., 
                delta_converged=1e-2, 
                max_steps=100, 
                batch_size=32):
        super().__init__()

        self.lr = lr
        self.regularization = regularization
        self.delta_converged = delta_converged 
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.W = None
        self.b = None

    def fit(self, X, y):
        X_, y_ = np.array(X), np.array(y)
        n_samples, n_features = X_.shape
        X_ = np.column_stack((np.ones(n_samples), X_))
        batch_split_mask = [x for x in range(self.batch_size, n_samples, self.batch_size)]
        weights = np.zeros((n_features + 1))
        num_steps = - 1

        while (num_steps := num_steps + 1) < self.max_steps:
            X_step, y_step = shuffle(X_, y_)
            X_batches, y_batches = np.split(X_step, batch_split_mask), np.split(y_step, batch_split_mask)
            for X_batch, y_batch in zip(X_batches, y_batches):
                curr_batch_len = len(y_batch)
                
                if curr_batch_len != self.batch_size:
                    continue
                reg_weights = np.copy(weights)
                reg_weights[0] = 0
                gradient = 2 * (X_batch.T @ (X_batch @ weights - y_batch))/self.batch_size  + (2 * self.regularization) * reg_weights
                weights -= self.lr * gradient
                if (np.linalg.norm(gradient) < self.delta_converged):
                    self.W = weights[1:]
                    self.b = weights[0]
                    return self
        
        self.W = weights[1:]
        self.b = weights[0]
        return self
                
    def predict(self, X):
        X_ = np.array(X)
        return X_ @ self.W + self.b
    

class OneHotPreprocessor(BaseDataPreprocessor):
    def __init__(self, categorical_columns: Optional[List[str]]=None, handle_unknown = 'ignore', **kwargs):
        super(OneHotPreprocessor, self).__init__(**kwargs)
        self.categorical_columns = categorical_columns
        self.one_hot_encoder = OneHotEncoder(handle_unknown = handle_unknown)
        
    def fit(self, data, *args):
        super().fit(data)
        self.one_hot_encoder.fit(data[self.categorical_columns])
        return self
        
    def transform(self, data):
        continuous_data = super().transform(data)
        categorial_data = self.one_hot_encoder.transform(data[self.categorical_columns])
        return np.hstack([continuous_data, categorial_data.toarray()])


def make_ultimate_pipeline(reg = 0.259):
    continuous_columns = ['Lot_Frontage', 'Year_Built', 'Year_Remod_Add', 'Mas_Vnr_Area', 'BsmtFin_SF_1', 'BsmtFin_SF_2', 'Bsmt_Unf_SF', 'Total_Bsmt_SF', 'First_Flr_SF', 'Second_Flr_SF', 'Gr_Liv_Area', 'Bsmt_Full_Bath', 'Bsmt_Half_Bath', 'Full_Bath', 'Half_Bath', 'Bedroom_AbvGr', 'Kitchen_AbvGr', 'TotRms_AbvGrd', 'Fireplaces', 'Garage_Cars', 'Garage_Area', 'Wood_Deck_SF', 'Open_Porch_SF', 'Enclosed_Porch', 'Screen_Porch', 'Mo_Sold', 'Year_Sold', 'Longitude', 'Latitude']
    categorical_columns = ['MS_SubClass', 'MS_Zoning', 'Street', 'Alley', 'Lot_Shape', 'Land_Contour', 'Utilities', 'Lot_Config', 'Land_Slope', 'Neighborhood', 'Condition_1', 'Condition_2', 'Bldg_Type', 'House_Style', 'Overall_Qual', 'Overall_Cond', 'Roof_Style', 'Roof_Matl', 'Exterior_1st', 'Exterior_2nd', 'Mas_Vnr_Type', 'Exter_Qual', 'Exter_Cond', 'Foundation', 'Bsmt_Qual', 'Bsmt_Cond', 'Bsmt_Exposure', 'BsmtFin_Type_1', 'BsmtFin_Type_2', 'Heating', 'Heating_QC', 'Central_Air', 'Electrical', 'Kitchen_Qual', 'Functional', 'Fireplace_Qu', 'Garage_Type', 'Garage_Finish', 'Garage_Qual', 'Garage_Cond', 'Paved_Drive', 'Pool_QC', 'Fence', 'Misc_Feature', 'Sale_Type', 'Sale_Condition']

    pipe =  Pipeline([('encoder', OneHotPreprocessor(continuous_columns = continuous_columns, categorical_columns = categorical_columns)), ('sgd_lin_reg', SGDLinearRegressor(regularization = reg, n_continuous_features = len(continuous_columns)))])
    return pipe