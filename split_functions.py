from __future__ import annotations
from typing import Protocol, NamedTuple, Any
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


from numba import njit
import numpy as np
from typing import Any
import matplotlib.pyplot as plt




# call some different strings to the splitting class
class MultiKeyDict:
    def __init__(self):
        self._store = {}
        self._key_map = {}
    
    def add(self, keys, value):
        """
        Adds a value to the dictionary with multiple keys.
        
        Args:
        keys (list or tuple): A list or tuple of keys that will map to the value.
        value: The value to be stored.
        """
        if not isinstance(keys, (list, tuple)):
            raise TypeError("Keys must be a list or tuple of strings")
        
        main_key = keys[0]
        self._store[main_key] = value
        
        for key in keys:
            self._key_map[key] = main_key
    
    def __getitem__(self, key):
        """
        Retrieves the value associated with the given key.
        
        Args:
        key (str): The key to look up.
        
        Returns:
        The value associated with the key.
        """
        if key in self._key_map:
            main_key = self._key_map[key]
            return self._store[main_key]
        else:
            raise KeyError(f"Key '{key}' not found.")
    
    def __repr__(self):
        return repr(self._store)
    
    def keys(self):
        return self._store.keys()
    
    def values(self):
        return self._store.values()
    
    
    
@njit
def repeat_array_as_matrix(array, num_rows):
    n = array.size
    matrix = np.zeros((num_rows, n), dtype=array.dtype)
    for i in range(num_rows):
        matrix[i, :] = array
    return matrix

@njit(cache=True)
def make_rand_vector(df:int,
                     dimensions:int) -> npt.NDArray[np.float64]:
    """
    Generate a random unitary vector in the unit ball with a maximum number of dimensions. 
    This vector will be successively used in the generation of the splitting hyperplanes.
    
    Args:
        df: Degrees of freedom
        dimensions: number of dimensions of the feature space

    Returns:
        vec: Random unitary vector in the unit ball
        
    """
    if dimensions<df:
        raise ValueError("degree of freedom does not match with dataset dimensions")
    else:
        vec_ = np.random.normal(loc=0.0, scale=1.0, size=df)
        indexes = np.random.choice(np.arange(dimensions),df,replace=False)
        vec = np.zeros(dimensions)
        vec[indexes] = vec_
        vec=vec/np.linalg.norm(vec)
    return vec


# initialize the protocol class of the splitting functions
class SplittingFunction:
    def __init__(self, *args, **kwargs):
        self.parameters: np.ndarray = None
        self.treshold: float = None

    def initialize_parameters(self, X: np.ndarray, *args, **kwargs):
        self.parameters = self.generate_parameters(X, *args, **kwargs)
    
    def initialize_treshold(self, distribution: np.ndarray, plus: bool = True):
        self.treshold = self.treshold_calculation(distribution, plus)

    @staticmethod
    def function(self, X: np.ndarray, parameters: np.ndarray) -> np.array:
        pass
    
    @staticmethod
    def generate_parameters(self, X: np.ndarray, *args, **kwargs) -> np.array:
        pass
    
    @staticmethod
    def treshold_calculation(self, distribution: np.ndarray, plus:bool) -> float:
        pass
    
    @staticmethod
    @njit
    def Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        pass
    
    def __call__(self, X: np.ndarray, parameters: np.ndarray) -> np.array:
        return self.function(X,parameters)
      
class Hyperplane(SplittingFunction):

    @staticmethod
    @njit
    def generate_parameters(X: np.ndarray, locked_dims:int=0) -> np.ndarray:
        #randomly generate a unitary vector
        parameters = make_rand_vector(X.shape[1]-locked_dims,X.shape[1])[:,None].T
        return parameters

    @staticmethod
    @njit
    def function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        parameters = parameters[0,:X.shape[1]]
        distribution = np.sum(X*parameters,axis=1)
        return distribution
    
    @staticmethod
    @njit
    def treshold_calculation(distribution: np.ndarray, plus:bool) -> float:
        if plus:
            treshold_value = np.random.normal(np.mean(distribution), np.std(distribution))
        else:
            treshold_value = np.random.uniform(np.min(distribution),np.max(distribution))
        return treshold_value
    
    @staticmethod
    @njit
    def Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        parameters = parameters[0,:X.shape[1]]
        return repeat_array_as_matrix(parameters, X.shape[0])

class CircleSplitting(SplittingFunction):
    @staticmethod
    def generate_parameters(X: np.ndarray, *args) -> np.ndarray:
        #randomly generate the center of a hypersphere inside the data distribution
        center = np.random.uniform(low=np.min(X, axis=0)-0.1, high=np.max(X, axis=0)+0.1)[:,None].T
        return center
    
    @staticmethod
    def function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        #calculate the distance of each point to the center of the hypersphere
        parameters = parameters[0,:X.shape[1]]
        distribution = np.linalg.norm(X - parameters, axis=1)
        return distribution
    
    @staticmethod
    def treshold_calculation(distribution, plus:bool) -> float:
        if plus:
            treshold_value = np.random.normal(np.mean(distribution), np.std(distribution))
        else:
            treshold_value = np.random.uniform(np.min(distribution),np.max(distribution))
        return treshold_value
    
    @staticmethod
    def Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        parameters = parameters[0,:X.shape[1]]
        return (X - parameters) / np.linalg.norm(X - parameters, axis=1)[:, None]
    

class SingleDimensionHyperplane(SplittingFunction):
    @staticmethod
    @njit
    def generate_parameters(X: np.ndarray, *args) -> np.ndarray:
        #randomly generate a unitary vector
        parameters = make_rand_vector(1,X.shape[1])[:,None].T
        return parameters
    
    @staticmethod
    @njit
    def function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        parameters = parameters[0,:X.shape[1]]
        distribution = np.sum(X*parameters,axis=1)
        return distribution
    
    @staticmethod
    @njit
    def treshold_calculation(distribution, plus:bool) -> float:
        if plus:
            treshold_value = np.random.normal(np.mean(distribution), np.std(distribution))
        else:
            treshold_value = np.random.uniform(np.min(distribution),np.max(distribution))
        return treshold_value
    
    @staticmethod
    @njit
    def Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        parameters = parameters[0,:X.shape[1]]
        return repeat_array_as_matrix(parameters, X.shape[0])
    
class X2MinusSinX1Splitting(SplittingFunction):
    @staticmethod
    def generate_parameters(X: np.ndarray, *args) -> np.ndarray:
        #randomly generate a unitary vector
        parameters = np.random.randn(1,3)
        return parameters
    
    @staticmethod
    def function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        distribution = X[:,1]-np.sin(X[:,0])
        return distribution
    
    @staticmethod
    def treshold_calculation(distribution, plus:bool) -> float:
        if plus:
            treshold_value = np.random.normal(np.mean(distribution), np.std(distribution)*2)
        else:
            treshold_value = np.random.uniform(np.min(distribution),np.max(distribution))
        return treshold_value
    
    @staticmethod
    def Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        return np.array([-np.cos(X[:,0]),np.ones(X.shape[0])]).T
    
class HyperbolicSplitting(SplittingFunction):
    @staticmethod
    def generate_parameters(X: np.ndarray, *args) -> np.ndarray:
        #randomly generate the center of a hypersphere inside the data distribution
        center_1 = np.random.uniform(low=np.min(X, axis=0)-0.1, high=np.max(X, axis=0)+0.1)
        center_2 = np.random.uniform(low=np.min(X, axis=0)-0.1, high=np.max(X, axis=0)+0.1)
        return np.array([center_1,center_2]).flatten()[:,None].T
    
    @staticmethod
    def function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        #calculate the distance of each point to the center of the hypersphere
        center_1 = parameters[0,:X.shape[1]]
        center_2 = parameters[0,X.shape[1]:2*X.shape[1]]
        distribution = np.linalg.norm(X - center_1, axis=1) - np.linalg.norm(X - center_2, axis=1)
        return distribution
    
    @staticmethod
    def treshold_calculation(distribution, plus:bool) -> float:
        if plus:
            treshold_value = np.random.normal(np.mean(distribution), np.std(distribution))
        else:
            treshold_value = np.random.uniform(np.min(distribution),np.max(distribution))
        return treshold_value
    
    @staticmethod
    def Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        center_1 = parameters[0,:X.shape[1]]
        center_2 = parameters[0,X.shape[1]:2*X.shape[1]]
        return (X - center_1) / np.linalg.norm(X - center_1, axis=1)[:, None] - (X - center_2) / np.linalg.norm(X - center_2, axis=1)[:, None]


class EllipsoidSplitting(SplittingFunction):
    @staticmethod
    def generate_parameters(X: np.ndarray, *args) -> np.ndarray:
        center_1 = np.random.uniform(low=np.min(X, axis=0)-0.1, high=np.max(X, axis=0)+0.1)
        center_2 = np.random.uniform(low=np.min(X, axis=0)-0.1, high=np.max(X, axis=0)+0.1)
        return np.array([center_1,center_2]).flatten()[:,None].T
    
    @staticmethod
    def function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        center_1 = parameters[0,:X.shape[1]]
        center_2 = parameters[0,X.shape[1]:2*X.shape[1]]
        distribution = np.linalg.norm(X - center_1, axis=1) + np.linalg.norm(X - center_2, axis=1)
        return distribution
    
    @staticmethod
    def treshold_calculation(distribution, plus:bool) -> float:
        if plus:
            treshold_value = np.random.normal(np.mean(distribution), np.std(distribution))
        else:
            treshold_value = np.random.uniform(np.min(distribution),np.max(distribution))
        return treshold_value
    
    @staticmethod
    def Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        center_1 = parameters[0,:X.shape[1]]
        center_2 = parameters[0,X.shape[1]:2*X.shape[1]]
        return (X - center_1) / np.linalg.norm(X - center_1, axis=1)[:, None] + (X - center_2) / np.linalg.norm(X - center_2, axis=1)[:, None]




class ParaboloidSplitting(SplittingFunction):

    @staticmethod
    def generate_parameters(X: np.ndarray, *args) -> np.ndarray:
        # Randomly generate the center of the paraboloid within the data distribution
        center = np.random.uniform(low=np.min(X, axis=0)-0.1, high=np.max(X, axis=0)+0.1)
        
        # Generate a random unitary vector for the normal
        normal = make_rand_vector(X.shape[1], X.shape[1])
        
        # Return center and normal vector combined as parameters
        return np.hstack((center, normal)).reshape(1, -1)
    
    @staticmethod
    def function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        center = parameters[0, :X.shape[1]]
        normal = parameters[0, X.shape[1]:2*X.shape[1]]

        # Calculate the paraboloid value
        distances = X - center
        distribution = np.linalg.norm(distances,axis = 1) + np.dot(X, normal)
        return distribution
    
    @staticmethod
    def treshold_calculation(distribution: np.ndarray, plus: bool) -> float:
        if plus:
            treshold_value = np.random.normal(np.mean(distribution), np.std(distribution))
        else:
            treshold_value = np.random.uniform(np.min(distribution), np.max(distribution))
        return treshold_value
    
    @staticmethod
    def Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        center = parameters[0, :X.shape[1]]
        normal = parameters[0, X.shape[1]:2*X.shape[1]]

        distances = X - center
        jacobian = 2 * distances + normal
        return jacobian
    
class ConicSplitting(SplittingFunction):
    @staticmethod
    def generate_parameters(X: np.ndarray, *args) -> np.ndarray:
        n = X.shape[1]
        # Generate a random symmetric matrix A with parameters in minX,maxX
        A = np.random.normal(0,1, size=(n, n))
        A = (A + A.T) / 2
        
        # Generate a random vector v
        v = np.random.uniform(-100,100, size=n)
        
        # Return A and v combined as parameters
        return np.hstack((A.flatten(), v)).reshape(1, -1)
    
    @staticmethod
    def function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        n = X.shape[1]
        A = parameters[0, :n*n].reshape(n, n)
        v = parameters[0, n*n:n*n+n]
        
        # Calculate the quadratic and linear terms
        quadratic_term = np.einsum('ij,jk,ik->i', X, A, X)
        linear_term = np.dot(X, v)
        distribution = quadratic_term + linear_term
        return distribution
    
    @staticmethod
    def treshold_calculation(distribution: np.ndarray, plus: bool) -> float:
        if plus:
            treshold_value = np.random.normal(np.mean(distribution), np.std(distribution))
        else:
            treshold_value = np.random.uniform(np.min(distribution), np.max(distribution))
        return treshold_value
    
    @staticmethod
    def Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        n = X.shape[1]
        A = parameters[0, :n*n].reshape(n, n)
        v = parameters[0, n*n:n*n+n]
        
        # Calculate the gradient
        gradient = (A + A.T) @ X.T + v[:, None]
        return gradient.T
        

    



class RandomNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, initialize=0):
        super(RandomNN, self).__init__()
        torch.manual_seed(initialize)  # Set the seed to a specific value
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        #x = torch.sigmoid(x)
        return x
    
    def predict(self, x):
        x = torch.from_numpy(x).float()
        with torch.no_grad():
            return self.forward(x).numpy().flatten()

class NNSplitting(SplittingFunction):
    @staticmethod
    def function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        n = X.shape[1]
        seed = int(parameters[0,0])

        model = RandomNN(n,n*3,initialize=seed)
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float()
            outputs = model(X_tensor)
            return outputs.numpy().flatten()

    @staticmethod
    def generate_parameters(X: np.ndarray, *args, **kwargs):
        # Parameters are internally managed by the neural network
        seed = np.random.randint(0,100000)
        seed = np.array([[seed]])
        return seed


    @staticmethod
    def treshold_calculation(distribution: np.ndarray, plus: bool) -> float:
        if plus:
            treshold_value = np.random.normal(np.mean(distribution), np.std(distribution))
        else:
            treshold_value = np.random.uniform(np.min(distribution), np.max(distribution))
        return treshold_value

    @staticmethod
    def Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        n = X.shape[1]
        seed = int(parameters[0,0])

        model = RandomNN(n,n*3,initialize=seed)
        
        X_tensor = torch.from_numpy(X).float().requires_grad_(True)
        outputs = model(X_tensor)

        # Compute the gradient of the single-dimensional output with respect to the input
        model.zero_grad()
        outputs.backward(torch.ones_like(outputs))

        gradient = X_tensor.grad

        # Convert gradient to numpy array if needed
        gradient_np = gradient.numpy()

        return gradient_np
    
    
    
splitting_functions_dictionary = MultiKeyDict()
splitting_functions_dictionary.add(['EIF', 'Hyperplane', 'Hyperplane_split'], Hyperplane)
splitting_functions_dictionary.add(['HIF', 'Hypersphere', 'Circle_split'], CircleSplitting)
splitting_functions_dictionary.add(['IF', 'SingleDimension', 'SingleDimension_split'], SingleDimensionHyperplane)
#splitting_functions_dictionary.add(['X2MinusSinX1', 'X2MinusSinX1_split'], X2MinusSinX1Splitting)
splitting_functions_dictionary.add(['HypIF','Hyperbolic', 'Hyperbolic_split'], HyperbolicSplitting)
splitting_functions_dictionary.add(['ParabIF', 'Paraboloid', 'Paraboloid_split'], ParaboloidSplitting)
splitting_functions_dictionary.add(['ConicIF', 'Conic', 'Conic_split'], ConicSplitting)
splitting_functions_dictionary.add(['EllIF', 'Ellipsoid', 'Ellipsoid_split'], EllipsoidSplitting)
#splitting_functions_dictionary.add(['NNIF', 'NN', 'NN_split'], NNSplitting)