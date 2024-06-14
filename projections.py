import numpy as np
from numpy import typing as npt
import random 

from typing import Optional
from typing import Protocol


import numpy as np
from numba import njit
from typing import Optional, Tuple
from numba import njit
from numba import prange
import numpy as np
from numba.typed import List
from numba.np.ufunc import parallel

# Define a simplified version of the transformation functions using njit
@njit
def apply_circular_inversion(x: np.ndarray, t: np.ndarray, d: np.ndarray) -> np.ndarray:
    if x.ndim <=1:
        raise ValueError("Input array must have 2 dimensions")
    if x.ndim > 2:
        raise ValueError("Input array must have 2 dimensions")
    if t is None:
        t = np.zeros_like(x)
    if d is None:
        d = np.ones_like(x)

    # if t.shape[0] == 1:
    #     t = np.tile(t, (x.shape[0], 1)) 
    
    numerator = (x - t) * (d ** 2)
    denominator = ((x - t) ** 2).sum(axis=1)[:, None]
    return numerator / denominator + t

@njit
def apply_translation(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    if t is None:
        t = np.zeros_like(x)
    return x + t

@njit
def apply_dilatation(x: np.ndarray, d: np.ndarray) -> np.ndarray:
    if d is None:
        d = np.ones_like(x)
    return x * d

@njit
def apply_identity(x: np.ndarray) -> np.ndarray:
    return x


@njit
def random_parameters(dim: int, n:int=4, center_interval = (0,1), radius_meanchi = 40):
    
    t_param = np.zeros((n, dim))
    d_param = np.zeros((n, dim))
    # Generate the random parameters using Numba's random functions
    for i in range(n):
        t_param[i, :] = np.random.choice(np.array([-1,1]), size=(1, dim))*np.random.uniform(center_interval[0],center_interval[1],size=(1, dim))
        d_param[i, :] = radius_meanchi#np.random.chisquare(radius_meanchi , (1, dim))
    
    return t_param, d_param

@njit
def concatenation_of_transformation(X:np.ndarray, t_parameters: np.ndarray, d_parameters: np.ndarray, type_transformation: Optional[np.ndarray]=None):
    if type_transformation is None:
        type_transformation = np.zeros(t_parameters.shape[0])
    
    for i in range(t_parameters.shape[0]):
        if type_transformation[i] == 0:
            X = apply_circular_inversion(X, t_parameters[i:i+1,:], d_parameters[i:i+1,:])
        elif type_transformation[i] == 1:
            X = apply_translation(X, t_parameters[i:i+1,:])
        elif type_transformation[i] == 2:
            X = apply_dilatation(X, d_parameters[i:i+1,:])
        elif type_transformation[i] == 3:
            X = apply_identity(X)
    
    return X
@njit
def concatenation_of_inverse_transformation(X:np.ndarray, t_parameters: np.ndarray, d_parameters: np.ndarray, type_transformation: Optional[np.ndarray]=None):
    if type_transformation is None:
        type_transformation = np.zeros(t_parameters.shape[0])
    
    s = t_parameters.shape[0]
    for i in range(s):
        if type_transformation[i] == 0:
            X = apply_circular_inversion(X, t_parameters[s-i-1:s-i,:], d_parameters[s-i-1:s-i,:])
        elif type_transformation[i] == 1:
            X = apply_translation(X, -t_parameters[s-i-1:s-i,:])
        elif type_transformation[i] == 2:
            X = apply_dilatation(X, 1/d_parameters[s-i-1:s-i,:])
        elif type_transformation[i] == 3:
            X = apply_identity(X)
    
    return X

# Random transformation creation
class RandomTransformations:
    def __init__(self, dim: int, len_transformations: int = 2, seed: Optional[int] = None, identity_rate: float = 0.0):
        self.dim = dim
        self.len_transformations = len_transformations
        self.seed = seed
        self.identity_rate = identity_rate
        self.t_parameters = np.zeros((self.len_transformations, self.dim))
        self.d_parameters = np.zeros((self.len_transformations, self.dim))
        self.transformations, self.inverse_transformations = None, None
        
        if seed is not None:
            np.random.seed(seed)

    
    def __repr__(self):
        return f"RandomTransformations(dim={self.dim}, len_transformations={self.len_transformations}, seed={self.seed}, identity_rate={self.identity_rate})"
    
    def parameters_setup(self, t_parameters: Optional[np.ndarray] = None, d_parameters: Optional[np.ndarray] = None):
        if t_parameters is None and d_parameters is None:
            for i in range(self.len_transformations):
                t_param, d_param = random_parameters(self.dim)
                self.t_parameters[i, :] = t_param
                self.d_parameters[i, :] = d_param
        elif t_parameters is not None and d_parameters is not None:
            self.t_parameters = t_parameters
            self.d_parameters = d_parameters
        else:
            raise ValueError("Both t_parameters and d_parameters must be provided or None")
        self.transformations, self.inverse_transformations = self.random_transformation()

    def random_transformation(self):
        transformations = []
        inverse_transformations = []

        for indx in range(self.len_transformations):
            t_param, d_param = self.t_parameters[indx,:], self.d_parameters[indx,:]
            func_idx = 0 #modifica per generalizzare le trasformazioni
            transformations.append((func_idx, t_param, d_param))

        if np.random.random() < self.identity_rate:
            transformations = [(3, None, None)] * self.len_transformations

        inverse_transformations = list(reversed(transformations))
        return transformations, inverse_transformations

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for func_idx, t, d in self.transformations:
            x = self.apply_transformation(x, func_idx, t, d)
        return x

    def inverse(self, x: np.ndarray) -> np.ndarray:
        for func_idx, t, d in self.inverse_transformations:
            x = self.apply_transformation(x, func_idx, t, d)
        return x

    def apply_transformation(self, x: np.ndarray, func_idx: int, t: np.ndarray, d: np.ndarray) -> np.ndarray:
        if func_idx == 0:
            return apply_circular_inversion(x, t, d)
        elif func_idx == 1:
            return apply_translation(x, t)
        elif func_idx == 2:
            return apply_dilatation(x, d)
        elif func_idx == 3:
            return apply_identity(x)
  
        
