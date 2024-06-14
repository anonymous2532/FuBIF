from __future__ import annotations

from typing import ClassVar, Optional, List, Union,Protocol, NamedTuple, Any
import numpy.typing as npt
from dataclasses import dataclass, field


import numpy as np

import torch
from numba import njit, prange, float64, int64, boolean
from numba.experimental import jitclass
from joblib import Parallel, delayed
from numba.typed import List

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from split_functions_optimized import *


import warnings

name = None


# Convert specific warnings into exceptions
warnings.filterwarnings('error', category=RuntimeWarning, message='Mean of empty slice.')

@njit
def safe_mean(arr):
    if arr.shape[0] == 0:
        return np.zeros(arr.shape[1])
    elif arr.shape[1] == 1:
        return np.array([arr.mean()])
    else:
        res = np.zeros((arr.shape[1],))
        for i in prange(arr.shape[1]):
            res[i] = np.mean(arr[:,i])
        return res

    
@njit(cache=True)
def c_factor(n: int) -> float:
    """
    Average path length of unsuccesful search in a binary search tree given n points.
    This is a constant factor that will be used as a normalization factor in the Anomaly Score calculation.
    
    Args:
        n: Number of data points for the BST.
        
    Returns:
        Average path length of unsuccesful search in a BST
        
    """
    if n <=1: return 0
    return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))


@njit
def get_leaf_ids(X, child_left, child_right, splitting_parameters, treshold)->np.array:
    leaf_ids = np.zeros(X.shape[0], dtype=np.int32)
    for i in range(X.shape[0]):
        x = X[i:i+1,:]
        node_id = 0
        while child_left[node_id] or child_right[node_id]:
            #scaling the data with respect to the node
            
            #apply the splitting function
            dist = splitting_function(x, splitting_parameters[node_id][:,None].T)
            d = dist - treshold[node_id]
            node_id = child_left[node_id] if d <= 0 else child_right[node_id]
        leaf_ids[i] = node_id
    return leaf_ids

# @njit
# def get_leaf_ids(X, child_left, child_right, splitting_functions_id, splitting_parameters, treshold, scaler_means, scaler_stds)->np.array:
#     leaf_ids = np.zeros(X.shape[0], dtype=np.int32)
#     for i in range(X.shape[0]):
#         x = X[i:i+1,:]
#         node_id = 0
#         while child_left[node_id] or child_right[node_id]:
#             #scaling the data with respect to the node
            
#             mean = scaler_means[node_id]
#             scale = scaler_stds[node_id]
#     #        x = (x - mean)
            
#             #apply the splitting function
#             func_id = splitting_functions_id[node_id]
#             splitting_function = splitting_functions[func_id]
#             dist = splitting_function.function(x, splitting_parameters[node_id][:,None].T)
#             d = dist - treshold[node_id]
#             node_id = child_left[node_id] if d <= 0 else child_right[node_id]
#         leaf_ids[i] = node_id
#     return leaf_ids

# def get_leaf_ids(X, child_left, child_right, splitting_functions_id, splitting_parameters, treshold, scaler_means, scaler_stds):
#     n_samples, n_features = X.shape
#     leaf_ids = np.zeros(n_samples, dtype=np.int32)
#     node_ids = np.zeros(n_samples, dtype=np.int32)

#     active_nodes = np.ones(n_samples, dtype=np.bool_)

#     while np.any(active_nodes):
#         current_nodes = node_ids[active_nodes]
#         current_samples = X[active_nodes]

#         # Apply scaling
#         means = scaler_means[current_nodes]
#         scales = scaler_stds[current_nodes].reshape(-1,1)
#         current_samples = (current_samples - means)
#         current_samples= current_samples/ scales

#         # Apply splitting functions
#         func_ids = splitting_functions_id[current_nodes]
#         if len(splitting_functions)==1:
#             dists = splitting_functions[0].function(current_samples, splitting_parameters[current_nodes])
#         else:
#             dists = np.zeros_like(current_samples[:, 0])
#             for i in range(len(current_samples)):
#                 func_id = func_ids[i]
#                 splitting_function = splitting_functions[func_id]
#                 dists[i] = splitting_function.function(current_samples[i], splitting_parameters[current_nodes[i]])

#         dists = dists - treshold[current_nodes]
        

#         # Determine next nodes
#         go_left = dists <= 0
#         go_right = ~go_left
#         import ipdb; ipdb.set_trace()
#         current_nodes[go_left] = child_left[current_nodes[go_left]]
#         current_nodes[go_right] = child_right[current_nodes[go_right]]

#         # Update active nodes (nodes that are not leaves)
#         active_nodes = (child_left[node_ids] != 0) & (child_right[node_ids] != 0)
    
#     leaf_ids = node_ids

#     return leaf_ids

@njit(cache=True)
def calculate_importances(paths:np.ndarray,
                          directions:np.ndarray, 
                          importances_left:np.ndarray, 
                          importances_right:np.ndarray, 
                          d:int) -> tuple[np.array,np.array]:
    
    """
    Calculate the importances of the features for the given paths and directions.

    Args:
        paths: Paths to the leaf nodes
        directions: Directions to the leaf nodes
        importances_left: Importances of the left child nodes
        importances_right: Importances of the right child nodes
        normals: Normal vectors of the splitting hyperplanes
        d: Number of dimensions in the dataset

    Returns:
        Importances of the features for the given paths and directions and the normal vectors.
    """

    # Flatten the paths and directions for 1D boolean indexing
    paths_flat = paths.flatten()
    directions_flat = directions.flatten()
    
    # Create masks for left and right directions
    left_mask_flat = directions_flat == -1
    right_mask_flat = directions_flat == 1
    
    # Use masks to filter flattened paths; initialize with -1 (or suitable default)
    left_paths_flat = np.full_like(paths_flat, -1)
    right_paths_flat = np.full_like(paths_flat, -1)
    
    # Apply the masks
    left_paths_flat[left_mask_flat] = paths_flat[left_mask_flat]
    right_paths_flat[right_mask_flat] = paths_flat[right_mask_flat]
    
    # Since importances are mentioned to be arrays of arrays, let's assume we can index them directly with the flattened paths
    # Note: This step might need adjustment based on the actual structure and intended calculations
    importances_sum_left = np.zeros((paths.shape[0],d), dtype=np.float64)  # Initialize to match number of rows in paths
    importances_sum_right = np.zeros((paths.shape[0],d), dtype=np.float64)
    
    #normals_sum = np.zeros((paths.shape[0],d), dtype=np.float64)  # Initialize to match number of rows in paths
    
    importances_sum_left = importances_left[left_paths_flat].reshape(paths.shape[0],paths.shape[1],d).sum(axis=1)
    importances_sum_right = importances_right[right_paths_flat].reshape(paths.shape[0],paths.shape[1],d).sum(axis=1)
    #normals_sum = np.abs(normals[paths_flat]).reshape(paths.shape[0],paths.shape[1],d).sum(axis=1) <-- proviamo a fare senza normalizzazione nell'idea che prendendo valori random ad ogni split le cose si ricalibrano

    importances_sum = importances_sum_left + importances_sum_right
    
    return importances_sum #, normals_sum




tree_spec = [
    ('plus', boolean),
    ("max_depth", int64),
    ("min_sample", int64),
    ("n", int64),
    ("d", int64),
    ("size", int64),
    ("node_count", int64),
    ("max_nodes", int64),
    ("path_to", int64[:,:]),
    ("path_to_Right_Left", int64[:,:]),
    ("child_left", int64[:]),
    ("child_right", int64[:]),
    ("splitting_parameters", float64[:,:]),
    ("treshold", float64[:]),
    ("node_size", int64[:]),
    ("depth", int64[:]),
    ("corrected_depth", float64[:]),
    ("importances_right", float64[:,:]),
    ("importances_left", float64[:,:]),
    ("eta", float64),
]

@jitclass(tree_spec)
class ExtendedTree:

    """
    Class that represents an Isolation Tree in the Extended Isolation Forest model.

     
    Attributes:
        plus (bool): Boolean flag to indicate if the model is a `EIF` or `EIF+`. Defaults to True (i.e. `EIF+`)
        locked_dims (int): Number of dimensions to be locked in the model. Defaults to 0
        max_depth (int): Maximum depth of the tree
        min_sample (int): Minimum number of samples in a node. Defaults to 1
        n (int): Number of samples in the dataset
        d (int): Number of dimensions in the dataset
        node_count (int): Counter for the number of nodes in the tree
        max_nodes (int): Maximum number of nodes in the tree. Defaults to 10000
        path_to (np.array): Array to store the path to the leaf nodes
        path_to_Right_Left (np.array): Array to store the path to the leaf nodes with directions
        child_left (np.array): Array to store the left child nodes
        child_right (np.array): Array to store the right child nodes
        normals (np.array): Array to store the normal vectors of the splitting hyperplanes
        intercepts (np.array): Array to store the intercept values of the splitting hyperplanes
        node_size (np.array): Array to store the size of the nodes
        depth (np.array): Array to store the depth of the nodes
        corrected_depth (np.array): Array to store the corrected depth of the nodes
        importances_right (np.array): Array to store the importances of the right child nodes
        importances_left (np.array): Array to store the importances of the left child nodes
        eta (float): Eta value for the model. Defaults to 1.5

    """

    def __init__(self,
                 n: int,
                 d: int,
                 max_depth: int,
                 min_sample: int = 1,
                 plus: bool = True,
                 eta: float = 1.5) -> None: # in the kwargs we can pass the parameters of the splitting functions such as locked_dims, etc.

        self.plus = plus
        self.max_depth = max_depth
        self.min_sample = min_sample
        self.n = n
        self.d = d
        self.node_count = 1
        self.eta = eta

        # Initialize arrays with an initial size
        self.size = 1000
        
        self.path_to = -np.ones((self.size, self.max_depth + 1), dtype=np.int64)
        self.path_to_Right_Left = np.zeros((self.size, self.max_depth + 1), dtype=np.int64)
        
        self.child_left = np.zeros(self.size, dtype=np.int64)
        self.child_right = np.zeros(self.size, dtype=np.int64)
        
        self.splitting_parameters = np.zeros((self.size, self.size), dtype=np.float64)
        self.treshold = np.zeros(self.size, dtype=np.float64)
        
        self.node_size = np.zeros(self.size, dtype=np.int64)
        
        self.depth = np.zeros(self.size, dtype=np.int64)
        self.corrected_depth = np.zeros(self.size, dtype=np.float64)
        
        self.importances_right = np.zeros((self.size, self.d), dtype=np.float64)
        self.importances_left = np.zeros((self.size, self.d), dtype=np.float64)




    def add_node(self) -> None:
        if self.node_count >= self.size:
            raise ValueError("The tree is full")
        self.node_count += 1
        
    def fit(self, X:np.array) -> None:
        """
        Fit the model to the dataset.

        Args:
            X: Input dataset

        Returns:
            The method fits the model and does not return any value.
        """

        self.path_to[0,0] = 0
        self.extend_tree(node_id=0, X=X, depth=0)
        self.corrected_depth = np.array([ (c_factor(k)+sum(path>-1))/c_factor(self.n) for i,(k,path) in enumerate(zip(self.node_size,self.path_to)) if i<self.node_count])

    def create_new_node(self, parent_id:int, direction:int) -> int:
        """
        Create a new node in the tree.

        Args:
            parent_id: Parent node id
            direction: Direction to the new node

        Returns:
            New node id

        """

        new_node_id = self.node_count
        self.add_node()
        self.path_to[new_node_id] = self.path_to[parent_id]
        self.path_to_Right_Left[new_node_id] = self.path_to_Right_Left[parent_id]
        self.path_to[new_node_id, self.depth[parent_id]+1] = new_node_id
        self.path_to_Right_Left[new_node_id, self.depth[parent_id]] = direction
        self.depth[new_node_id] = self.depth[parent_id]+1     
        return new_node_id
    
    def extend_tree(self,
                    node_id:int,
                    X:npt.NDArray,
                    depth: int) -> None:
        
        """
        Extend the tree to the given node.

        Args:
            node_id: Node id
            X: Input dataset
            depth: Depth of the node
        
        Returns:
            The method extends the tree and does not return any value.
        """
        stack = [(0, X, 0)] 
        
        while stack:
            node_id, data, depth = stack.pop()
            
            self.node_size[node_id] = len(data)
            if self.node_size[node_id] <= self.min_sample or depth >= self.max_depth:
                continue
            
            data_scaled = data
            
            #use the splitting function to split the data
            parameters = splitting_parameters_function(data_scaled)
            dist = splitting_function(data_scaled, parameters)
            m = treshold(dist, self.plus)
            Jacobian = splitting_derivative(data_scaled, parameters)
            Jacobian = Jacobian**2/np.linalg.norm(Jacobian)**2
            mask = dist <= m
            
            X_left, Jacobian_left = data[mask], Jacobian[mask]
            X_right, Jacobian_right = data[~mask], Jacobian[~mask]

            #save the splitting parameters
            self.splitting_parameters[node_id,:parameters.shape[1]] = parameters
            self.treshold[node_id] = m
            
            #calculate importances
            self.importances_left[node_id] = safe_mean(Jacobian_left)*(self.node_size[node_id]/(len(X_left)+1))
            self.importances_right[node_id] = safe_mean(Jacobian_right)*(self.node_size[node_id]/(len(X_right)+1))
            

            #initialize the new nodes
            left_child = self.create_new_node(node_id,-1)
            right_child = self.create_new_node(node_id,1)
            
            self.child_left[node_id] = left_child
            self.child_right[node_id] = right_child

            stack.append((left_child, X_left, depth + 1))
            stack.append((right_child, X_right, depth + 1))
            

    def leaf_ids(self, X:np.array) -> np.array:
        """
        Get the leaf node ids for each data point in the dataset.

        This is a stub method of `get_leaf_ids`.

        Args:
            X: Input dataset

        Returns:
           Leaf node ids for each data point in the dataset.
        """
        return get_leaf_ids(X, self.child_left, self.child_right, self.splitting_parameters, self.treshold)
                       
    def apply(self, X:np.array) -> None:
        """
        Update the `path_to` attribute with the path to the leaf nodes for each data point in the dataset.

        Args:
            X: Input dataset

        Returns:
            The method updates `path_to` and does not return any value.
        """
        return self.path_to[self.leaf_ids(X)] 
    
    def predict(self,X,ids:np.array) -> np.array:

        """
        Predict the anomaly score for each data point in the dataset.

        Args:
            ids: Leaf node ids for each data point in the dataset.

        Returns:
            Anomaly score for each data point in the dataset.
        """
        return self.corrected_depth[ids],
    
    def importances(self,ids:np.array) -> tuple[np.array,np.array]:

        """
        Compute the importances of the features for the given leaf node ids.

        Args:
            ids: Leaf node ids for each data point in the dataset.

        Returns:
            Importances of the features for the given leaf node ids and the normal vectors.
        """
        importances = calculate_importances(
            self.path_to[ids], 
            self.path_to_Right_Left[ids], 
            self.importances_left, 
            self.importances_right, 
            self.d
        )
        
        return importances


class ExtendedIsolationForest():

    """
    Class that represents the Extended Isolation Forest model.

    Attributes:
        n_estimators (int): Number of trees in the model. Defaults to 400
        max_samples (int): Maximum number of samples in a node. Defaults to 256
        max_depth (int): Maximum depth of the trees. Defaults to "auto"
        plus (bool): Boolean flag to indicate if the model is a `EIF` or `EIF+`.
        name (str): Name of the model
        ids (np.array): Leaf node ids for each data point in the dataset. Defaults to None
        X (np.array): Input dataset. Defaults to None
        eta (float): Eta value for the model. Defaults to 1.5
        avg_number_of_nodes (int): Average number of nodes in the trees
    
    """

    def __init__(self,
                 plus:bool,
                 n_estimators:int=400,
                 max_depth:Union[str,int]="auto",
                 max_samples:Union[str,int]="auto",
                 eta:float = 1.5) -> None:
        
        self.n_estimators = n_estimators
        self.max_samples = 256 if max_samples == "auto" else max_samples
        self.max_depth = max_depth
        self.plus=plus
        self.name=name
        self.ids=None
        self.X=None
        self.eta=eta
    
    def __repr__(self) -> str:
        return f"{self.name}(n_estimators={self.n_estimators}, max_depth={self.max_depth}, max_samples={self.max_samples}), plus={self.plus})"
        
        
    def fit(self, X:np.array) -> None:

        """
        Fit the model to the dataset.

        Args:
            X: Input dataset
            locked_dims: Number of dimensions to be locked in the model. Defaults to None

        Returns:
            The method fits the model and does not return any value.
        """

        self.ids = None

        if self.max_depth == "auto":
            self.max_depth = int(np.ceil(np.log2(self.max_samples)))
        subsample_size = np.min((self.max_samples, len(X)))
        self.trees = [ExtendedTree(subsample_size, X.shape[1], self.max_depth, plus=self.plus, eta=self.eta) for _ in range(self.n_estimators)]
        
        for Tree in self.trees:
            Tree.fit(X[np.random.randint(len(X), size=subsample_size)])
            
    def compute_ids(self, X:np.array) -> None:
        
        """
        Compute the leaf node ids for each data point in the dataset.

        Args:
            X: Input dataset

        Returns:
            The method computes the leaf node ids and does not return any value.
        """
        if self.ids is None or self.X.shape != X.shape:
            self.X = X
            self.ids = np.array([tree.leaf_ids(X) for tree in self.trees])

    def predict(self, X:np.array) -> np.array:

        """
        Predict the anomaly score for each data point in the dataset.

        Args:
            X: Input dataset

        Returns:
            Anomaly score for each data point in the dataset.
        """
        self.compute_ids(X)
        predictions=[tree.predict(X,self.ids[i]) for i,tree in enumerate(self.trees)]
        values = np.array([p[0] for p in predictions])
        return np.power(2,-np.mean([value for value in values], axis=0))
    
    def _predict(self,X:np.array,p:float) -> np.array:
        """
        Predict the class of each data point (i.e. inlier or outlier) based on the anomaly score.

        Args:
            X: Input dataset
            p: Proportion of outliers (i.e. threshold for the anomaly score)

        Returns:
           Class labels (i.e. 0 for inliers and 1 for outliers)
        """
        An_score = self.predict(X)
        y_hat = An_score > sorted(An_score,reverse=True)[int(p*len(An_score))]
        return y_hat

    def _importances(self,X:np.array,ids:np.array) -> tuple[np.array,np.array]:

        """
        Compute the importances of the features for the given leaf node ids.

        Args:
            X: Input dataset
            ids: Leaf node ids for each data point in the dataset.

        Returns:
            Importances of the features for the given leaf node ids and the normal vectors.

        """
        importances = np.zeros(X.shape)
        for i,T in enumerate(self.trees):
            importance = T.importances(ids[i])
            importances += importance
        return importances/self.n_estimators
    
    def global_importances(self,X:np.array,p:float=0.1) -> np.array:

        """
        Compute the global importances of the features for the dataset.

        Args:
            X: Input dataset
            p: Proportion of outliers (i.e. threshold for the anomaly score). Defaults to 0.1

        Returns:
            Global importances of the features for the dataset.
        """

        self.compute_ids(X)
        y_hat = self._predict(X,p)
        importances = self._importances(X, self.ids)
        outliers_importances = np.sum(importances[y_hat],axis=0)
        inliers_importances = np.sum(importances[~y_hat],axis=0)
        return outliers_importances/inliers_importances
    
    def local_importances(self,
                          X:np.array) -> np.array:

        """
        Compute the local importances of the features for the dataset.

        Args:
            X: Input dataset

        Returns:
           Local importances of the features for the dataset.
        """
        
        self.compute_ids(X)
        importances = self._importances(X, self.ids)
        return importances


