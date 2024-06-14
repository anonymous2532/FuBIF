from __future__ import annotations
from typing import Protocol, NamedTuple, Any
import numpy as np
import numpy.typing as npt



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
def min_axis_0(arr):
    n, m = arr.shape
    min_vals = np.empty(m)
    for j in range(m):
        min_val = arr[0, j]
        for i in range(1, n):
            if arr[i, j] < min_val:
                min_val = arr[i, j]
        min_vals[j] = min_val
    return min_vals 

@njit
def max_axis_0(arr):
    n, m = arr.shape
    max_vals = np.empty(m)
    for j in range(m):
        max_val = arr[0, j]
        for i in range(1, n):
            if arr[i, j] > max_val:
                max_val = arr[i, j]
        max_vals[j] = max_val
    return max_vals
    
    
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
    def __init__(self):
        self.parameters: np.ndarray = None
        self.treshold: float = None

    def initialize_parameters(self, X: np.ndarray):
        self.parameters = self.generate_parameters(X)
    
    def initialize_treshold(self, distribution: np.ndarray, plus: bool = True):
        self.treshold = self.treshold_calculation(distribution, plus)

    @staticmethod
    def function(self, X: np.ndarray, parameters: np.ndarray) -> np.array:
        pass
    
    @staticmethod
    def generate_parameters(self, X: np.ndarray) -> np.array:
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
    
    
    
    
    
@njit
def treshold_calculation(distribution: np.ndarray, plus:bool) -> float:
    if plus:
        treshold_value = np.random.normal(np.mean(distribution), np.std(distribution))
    else:
        treshold_value = np.random.uniform(np.min(distribution),np.max(distribution))
    return treshold_value
      
############################################################################################################
################################ hyperplane splitting function ############################################
############################################################################################################

@njit
def hyperplane_generate_parameters(X: np.ndarray, locked_dims:int=0) -> np.ndarray:
    #randomly generate a unitary vector
    parameters = make_rand_vector(X.shape[1]-locked_dims,X.shape[1])[:,None].T
    return parameters

@njit
def hyperplane_function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    parameters = parameters[0,:X.shape[1]]
    distribution = np.sum(X*parameters,axis=1)
    return distribution

@njit
def hyperplane_Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    parameters = parameters[0,:X.shape[1]]
    return repeat_array_as_matrix(parameters, X.shape[0])


############################################################################################################
################################ hypersphere splitting function ############################################
############################################################################################################

@njit
def hypersphere_generate_parameters(X: np.ndarray) -> np.ndarray:
    #randomly generate the center of a hypersphere inside the data distribution
    min_vals = min_axis_0(X) - 0.1
    max_vals = max_axis_0(X) + 0.1
    
    center = np.empty(X.shape[1])
    for i in range(X.shape[1]):
        center[i] = np.random.uniform(min_vals[i], max_vals[i])
    return center.reshape(1, -1)

@njit
def hypersphere_function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    #calculate the distance of each point to the center of the hypersphere
    parameters = parameters[0,:X.shape[1]]
    distribution = np.sqrt(np.sum((X-parameters)**2, axis=1))
    return distribution

@njit
def hypersphere_Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    parameters = parameters[0,:X.shape[1]]
    distance = X - parameters
    norm_distance = np.sqrt(np.sum(distance**2, axis=1))
    return distance / norm_distance[:, None]


############################################################################################################
################################ singledimension splitting function ############################################
############################################################################################################


@njit
def onedim_generate_parameters(X: np.ndarray) -> np.ndarray:
    #randomly generate a unitary vector
    parameters = make_rand_vector(1,X.shape[1])[:,None].T
    return parameters


@njit
def onedim_function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    parameters = parameters[0,:X.shape[1]]
    distribution = np.sum(X*parameters,axis=1)
    return distribution



@njit
def onedim_Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    parameters = parameters[0,:X.shape[1]]
    return repeat_array_as_matrix(parameters, X.shape[0])
 
############################################################################################################
################################ X2MinusSinX1 splitting function ############################################
############################################################################################################
 
@njit
def X2MinusSinX1_generate_parameters(X: np.ndarray) -> np.ndarray:
    #randomly generate a unitary vector
    parameters = np.random.randn(1,3)
    return parameters

@njit
def X2MinusSinX1_function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    distribution = X[:,1]-np.sin(X[:,0])
    return distribution

@njit
def X2MinusSinX1_Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    return np.array([-np.cos(X[:,0]),np.ones(X.shape[0])]).T

############################################################################################################
################################ Hyperbolic splitting function ############################################
############################################################################################################ 
    
@njit
def Hyperbolic_generate_parameters(X: np.ndarray) -> np.ndarray:
    min_vals = min_axis_0(X) - 0.1
    max_vals = max_axis_0(X) + 0.1
    
    center_1 = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        center_1[i] = np.random.uniform(min_vals[i], max_vals[i])
    
    center_2 = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        center_2[i] = np.random.uniform(min_vals[i], max_vals[i])
    
    parameters = np.zeros((1,2*X.shape[1]))
    parameters[0,:X.shape[1]] = center_1
    parameters[0,X.shape[1]:2*X.shape[1]] = center_2
    return parameters

@njit
def Hyperbolic_function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    #calculate the distance of each point to the center of the hypersphere
    center_1 = parameters[0,:X.shape[1]]
    center_2 = parameters[0,X.shape[1]:2*X.shape[1]]
    distances_1 = X - center_1
    distances_2 = X - center_2
    norm_distances_1 = np.sqrt(np.sum(distances_1**2, axis=1))
    norm_distances_2 = np.sqrt(np.sum(distances_2**2, axis=1))
    distribution = norm_distances_1 - norm_distances_2
    return distribution

@njit
def Hyperbolic_Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    center_1 = parameters[0,:X.shape[1]]
    center_2 = parameters[0,X.shape[1]:2*X.shape[1]]
    distances_1 = X - center_1
    distances_2 = X - center_2
    norm_distances_1 = np.sqrt(np.sum(distances_1**2, axis=1))
    norm_distances_2 = np.sqrt(np.sum(distances_2**2, axis=1))
    return distances_1 / norm_distances_1[:, None] - distances_2 / norm_distances_2[:, None]

############################################################################################################
################################ Ellipsoid splitting function ############################################
############################################################################################################ 

@njit
def Ellipsoid_generate_parameters(X: np.ndarray) -> np.ndarray:
    min_vals = min_axis_0(X) - 0.1
    max_vals = max_axis_0(X) + 0.1
    
    center_1 = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        center_1[i] = np.random.uniform(min_vals[i], max_vals[i])
    
    center_2 = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        center_2[i] = np.random.uniform(min_vals[i], max_vals[i])
    
    parameters = np.zeros((1,2*X.shape[1]))
    parameters[0,:X.shape[1]] = center_1
    parameters[0,X.shape[1]:2*X.shape[1]] = center_2
    return parameters

@njit
def Ellipsoid_function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    center_1 = parameters[0,:X.shape[1]]
    center_2 = parameters[0,X.shape[1]:2*X.shape[1]]
    distances_1 = X - center_1
    distances_2 = X - center_2
    norm_distances_1 = np.sqrt(np.sum(distances_1**2, axis=1))
    norm_distances_2 = np.sqrt(np.sum(distances_2**2, axis=1))
    distribution = norm_distances_1 + norm_distances_2
    return distribution

@njit
def Ellipsoid_Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    center_1 = parameters[0,:X.shape[1]]
    center_2 = parameters[0,X.shape[1]:2*X.shape[1]]
    distances_1 = X - center_1
    distances_2 = X - center_2
    norm_distances_1 = np.sqrt(np.sum(distances_1**2, axis=1))
    norm_distances_2 = np.sqrt(np.sum(distances_2**2, axis=1))
    return distances_1 / norm_distances_1[:, None] + distances_2 / norm_distances_2[:, None]

############################################################################################################
################################ Paraboloid splitting function ############################################
############################################################################################################ 

@njit
def Paraboloid_generate_parameters(X: np.ndarray) -> np.ndarray:
    min_vals = min_axis_0(X) - 0.1
    max_vals = max_axis_0(X) + 0.1
    
    center = np.empty(X.shape[1])
    for i in range(X.shape[1]):
        center[i] = np.random.uniform(min_vals[i], max_vals[i])
    
    normal = np.random.randn(X.shape[1])
    normal /= np.linalg.norm(normal)  # Normalize the vector
    parameters = np.concatenate((center, normal))
    # Return center and normal vector combined as parameters
    return parameters.reshape(1, -1)

@njit
def Paraboloid_function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    center = parameters[0, :X.shape[1]]
    normal = parameters[0, X.shape[1]:2*X.shape[1]]

    # Calculate the paraboloid value
    distances = X - center
    norm_distances = np.sqrt(np.sum(distances**2, axis=1))
    distribution = norm_distances + np.dot(np.ascontiguousarray(X), np.ascontiguousarray(normal))
    return distribution

@njit
def Paraboloid_Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    center = parameters[0, :X.shape[1]]
    normal = parameters[0, X.shape[1]:2*X.shape[1]]

    distances = X - center
    jacobian = 2 * distances + normal
    return jacobian

############################################################################################################
################################ Conic splitting function ############################################
############################################################################################################ 
    
# @njit
# def Conic_generate_parameters(X: np.ndarray) -> np.ndarray:
#     n = X.shape[1]
#     # Generate a random symmetric matrix A with parameters in minX,maxX
#     A = np.random.normal(0,1, size=(n, n))
#     A = (A + A.T) / 2
    
#     # Generate a random vector v
#     v = np.random.uniform(-100,100, size=n)
    
#     # Return A and v combined as parameters
#     return np.hstack((A.flatten(), v)).reshape(1, -1)

# @njit
# def Conic_function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
#     n = X.shape[1]
#     A = parameters[0, :n*n].reshape(n, n)
#     v = parameters[0, n*n:n*n+n]
    
#     # Calculate the quadratic and linear terms
#     quadratic_term = np.einsum('ij,jk,ik->i', X, A, X)
#     linear_term = np.dot(X, v)
#     distribution = quadratic_term + linear_term
#     return distribution

# @njit
# def Conic_Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
#     n = X.shape[1]
#     A = parameters[0, :n*n].reshape(n, n)
#     v = parameters[0, n*n:n*n+n]
    
#     # Calculate the gradient
#     gradient = (A + A.T) @ X.T + v[:, None]
#     return gradient.T

@njit
def Conic_generate_parameters(X: np.ndarray) -> np.ndarray:
    n = X.shape[1]
    # Generate a random symmetric matrix A with parameters in minX, maxX
    A = np.random.normal(0, 1, size=(n, n))
    A = (A + A.T) / 2
    
    # Generate a random vector v
    v = np.random.uniform(-100, 100, size=n)
    
    # Return A and v combined as parameters
    return np.concatenate((A.flatten(), v)).reshape(1, -1)

@njit
def Conic_function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    n = X.shape[1]
    parameters_contiguous = np.ascontiguousarray(parameters[0, :n*n])
    A = parameters_contiguous.reshape(n, n)
    v = parameters[0,n*n:n*n+n]
    
    # Calculate the quadratic term manually
    quadratic_term = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        for j in range(n):
            for k in range(n):
                quadratic_term[i] += X[i, j] * A[j, k] * X[i, k]
    
    # Calculate the linear term
    linear_term = np.dot(np.ascontiguousarray(X), np.ascontiguousarray(v))
    
    distribution = quadratic_term + linear_term
    return distribution

@njit
def Conic_Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    n = X.shape[1]
    parameters_contiguous = np.ascontiguousarray(parameters[0, :n*n])
    A = parameters_contiguous.reshape(n, n)

    v = parameters[0,n*n:n*n+n]
    
    # Calculate the gradient
    gradient = np.dot(A + A.T, np.ascontiguousarray(X.T)) + v[:, None]
    return gradient.T

###################################################################################################
################################ NN splitting function ############################################
################################################################################################### 

from numba import njit, prange
  
@njit
def generate_normal(size):
    result = np.empty(size)
    for i in prange(size[0]):
        for j in prange(size[1]):
            result[i, j] = np.random.randn()
    return result

@njit
def generate_uniform(size):
    result = np.empty(size)
    for i in prange(size[0]):
        for j in prange(size[1]):
            result[i, j] = np.random.uniform(0,10)
    return result
  
# Activation functions and their derivatives
@njit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@njit
def relu(x):
    return np.maximum(0, x)

@njit
def sigmoid_derivative(x):
    return x * (1 - x)


@njit
def relu_derivative(x):
    return np.where(x > 0, 1, 0)


@njit
def NN_generate_parameters(X: np.ndarray) -> np.ndarray:
    seed = np.random.randint(0,100000)
    seed = np.array([[seed]])
    return seed

@njit
def initialize_parameters(input_size, hidden_size, output_size,seed):
    set_seed(seed[0,0])
    W1 = generate_normal(size=(input_size, hidden_size))
    b1 = generate_uniform(size=(1, hidden_size))
    W2 = generate_normal(size=(hidden_size, output_size))
    b2 = generate_uniform(size=(1, output_size))
    return W1, b1, W2, b2

@njit
def set_seed(seed):
    np.random.seed(int(seed))
    
# Forward pass
@njit
def NN_function(X, seed):
    W1, b1, W2, b2 = initialize_parameters(X.shape[1], X.shape[1]*3, 1,seed)
    Z1 = np.dot(np.ascontiguousarray(X), np.ascontiguousarray(W1))  + b1
    A1 = relu(Z1)
    Z2 = np.dot(np.ascontiguousarray(A1), np.ascontiguousarray(W2)) + b2
    A2 = sigmoid(Z2)
    return A2.flatten()


# Compute gradient of the output with respect to the input
@njit
def NN_Jacobian(X, seed):
    W1, b1, W2, b2 = initialize_parameters(X.shape[1], X.shape[1]*3, 1, seed)
    Z1 = np.dot(np.ascontiguousarray(X), np.ascontiguousarray(W1)) + b1
    A1 = relu(Z1)
    Z2 = np.dot(np.ascontiguousarray(A1), np.ascontiguousarray(W2)) + b2
    A2 = sigmoid(Z2)
    
    
    m, n = X.shape
    dX = np.zeros((m, n))
    
    # Gradient of the output with respect to the output layer inputs

    dz2 = sigmoid_derivative(Z2)
    
    # Gradient of the output with respect to the hidden layer activations
    dz1 = np.ascontiguousarray(dz2).dot(np.ascontiguousarray(W2.T)) * relu_derivative(Z1)

    
    # Gradient of the output with respect to the inputs
    dX = np.ascontiguousarray(dz1).dot(np.ascontiguousarray(W1.T))
    
    return dX
    
    
###################################################################################################
################################ bisec2dim splitting function ############################################
################################################################################################### 

@njit
def bisec2dim_generate_parameters(X: np.ndarray) -> np.ndarray:
    #randomly generate a unitary vector
    np.random.rand(1,2)
    parameters = np.array([[1.,1-2*np.random.randint(0,2)]])/np.sqrt(2)
    return parameters


@njit
def bisec2dim_function(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    parameters = parameters[0,:X.shape[1]]
    distribution = np.sum(X*parameters,axis=1)
    return distribution



@njit
def bisec2dim_Jacobian(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    parameters = parameters[0,:X.shape[1]]
    return repeat_array_as_matrix(parameters, X.shape[0])