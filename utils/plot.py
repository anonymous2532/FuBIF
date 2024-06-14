import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import ClassVar, Optional, List, Union, Tuple
import numpy.typing as npt
import time


from past_works.EIF_reboot import *
from utils.datasets import *
from projections import *

def plot_everything(X,X_t):
        
    x, y = zip(*X)
    x_t, y_t = zip(*X_t)

    # Create a new figure
    fig, ax = plt.subplots()

    # Create a circle with radius 1 centered at (0, 0)
    circle = Circle((-0.5, -1), 1, edgecolor='blue', facecolor='none')

    # Add the circle to the plot
    ax.add_patch(circle)
    ax.scatter(x, y, color='green')  # Plot points in green
    ax.scatter(x_t, y_t, color='red') 

    # Set the aspect of the plot to be equal, so the circle isn't skewed
    ax.set_aspect('equal')

    # Set limits to make sure the circle is centered and fits well
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-2, 2)

    # Display the plot
   # plt.show()
   

def plot_the_cuts(I:ExtendedIsolationForest,
                  X:npt.NDArray,
                  tree:int = 0,
                  features:Tuple[int,int] = (0,1)):
    
    Tree = I.trees[tree]
    Transformation = I.transformations[tree]
    X_trans = Transformation(X)
    for x in X_trans:
        node_id = 0
        n = []
        p = []
        node = []
        while Tree.child_left[node_id] or Tree.child_right[node_id]:
            n_new = Tree.normals[node_id]
            p_new = Tree.intercepts[node_id]
            x_values = np.array(list(np.logspace(0, 10, 1000))+list(-np.logspace(0,10, 1000)))
            y_values = (-n_new[features[0]]*x_values + p_new)/n_new[features[1]]
            X_line = np.zeros((2000,X.shape[1]))
            X_line[:,features[0]] = x_values
            X_line[:,features[1]] = y_values
            if len(n)==0 and len(p)==0:
                None
            else:
                for num,_ in enumerate(n):
                    n_line = n[num][[features[0],features[1]]]
                    p_line = p[num]
                    X_line_dist = np.dot(np.ascontiguousarray(X_line[:,list(features)]),np.ascontiguousarray(n_line))
                    if node[num] == "L": 
                        X_line = X_line[X_line_dist<=p_line]
                    else:
                        X_line = X_line[X_line_dist>=p_line]
            X_line = Transformation.inverse(X_line)
            plt.plot(X_line[:,features[0]], X_line[:,features[1]], color='black')
            d = np.dot(np.ascontiguousarray(x),np.ascontiguousarray(n_new))
            if d <= p_new:
                node_id = Tree.child_left[node_id] 
                node.append("L")
            else:
                node_id = Tree.child_right[node_id] 
                node.append("R") 
            n.append(n_new)
            p.append(p_new)
    
    plt.scatter(X[:,features[0]], X[:,features[1]], color='green')
    x_min,x_max = np.min(X[:,features[0]]), np.max(X[:,features[0]])
    y_min,y_max = np.min(X[:,features[1]]), np.max(X[:,features[1]])
    plt.xlim(x_min-0.2*(x_max-x_min), x_max+0.2*(x_max-x_min))
    plt.ylim(y_min-0.2*(y_max-y_min), y_max+0.2*(y_max-y_min))
    plt.show()
     
def plot_score_map(I:ExtendedIsolationForest,
                   X:npt.NDArray,
                   features:Tuple[int,int] = (0,1),
                   resolution:int = 30,
                   path:str = None,
                   labels:Optional[Tuple] = None):
    
    t = time.localtime()
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", t)
    zoom = 6
    x_min,x_max = X[:,features[0]].min()-zoom,X[:,features[0]].max()+zoom
    y_min,y_max = X[:,features[1]].min()-zoom/4,X[:,features[1]].max()+zoom/4
    x = np.linspace(x_min-0.2*(x_max-x_min), x_max+0.2*(x_max-x_min), resolution)
    y = np.linspace(y_min-0.2*(y_max-y_min), y_max+0.2*(y_max-y_min), resolution)
    X_grid = np.meshgrid(x, y)
    X_grid = np.array(X_grid).reshape(2, -1).T
    X_pred = np.ones((int(resolution*resolution), X.shape[1]))
    X_pred = X.mean(axis=0) * X_pred
    X_pred[:,features[0]] = X_grid[:,0]
    X_pred[:,features[1]] = X_grid[:,1]
    I.ids = None
    Z = I.predict(X_pred)
    Z = Z.reshape(resolution, resolution)
    plt.contourf(x, y, Z, cmap="Reds")
    plt.colorbar()
    plt.contour(x, y, Z, cmap="Greys", linewidths=0.2)
    plt.scatter(X[:,features[0]], X[:,features[1]], color='green',edgecolors="black")
    plt.xlim(x_min-0.2*(x_max-x_min), x_max+0.2*(x_max-x_min))
    plt.ylim(y_min-0.2*(y_max-y_min), y_max+0.2*(y_max-y_min))
    if labels:
        plt.xlabel(str(labels[0]))
        plt.ylabel(str(labels[1]))
    else:
        plt.xlabel("Feature "+str(features[0]))
        plt.ylabel("Feature "+str(features[1]))
    if path:
        plt.savefig(path+current_time+"_"+I.__repr__()+"_score_map.png")
    plt.show()


def plot_the_cutsV2(I:ExtendedIsolationForest,
                  X:npt.NDArray,
                  tree:int = 0,
                  features:Tuple[int,int] = (0,1)):
    
    Tree = I.trees[tree]
    for x in X:
        node_id = 0
        n = []
        p = []
        node = []
        while Tree.child_left[node_id] or Tree.child_right[node_id]:
            n_new = Tree.normals[node_id]
            p_new = Tree.intercepts[node_id]
            
            x_values = np.array(list(np.logspace(0, 100, 1000))+list(-np.logspace(0,100, 1000)))
            y_values = (-n_new[features[0]]*x_values + p_new)/n_new[features[1]]
            X_line = np.zeros((2000,X.shape[1]))
            X_line[:,features[0]] = x_values
            X_line[:,features[1]] = y_values
            if len(n)==0 and len(p)==0:
                None
            else:
                for num,_ in enumerate(n):
                    n_line = n[num][[features[0],features[1]]]
                    p_line = p[num]
                    X_line_dist = np.dot(np.ascontiguousarray(X_line[:,list(features)]),np.ascontiguousarray(n_line))
                    if node[num] == "L": 
                        X_line = X_line[X_line_dist<=p_line]
                    else:
                        X_line = X_line[X_line_dist>=p_line]
            
            d_param = Tree.d_parameters[node_id]
            t_param = Tree.t_parameters[node_id]
            mean = Tree.scaler_means[node_id]
            std = Tree.scaler_stds[node_id]
            X_line = (X_line*std)+mean
            X_line = concatenation_of_inverse_transformation(X_line, t_param, d_param)
            plt.plot(X_line[:,features[0]], X_line[:,features[1]], color='black')
            d = np.dot(np.ascontiguousarray(x),np.ascontiguousarray(n_new))
            if d <= p_new:
                node_id = Tree.child_left[node_id] 
                node.append("L")
            else:
                node_id = Tree.child_right[node_id] 
                node.append("R") 
            n.append(n_new)
            p.append(p_new)
    
    plt.scatter(X[:,features[0]], X[:,features[1]], color='green')
    x_min,x_max = np.min(X[:,features[0]]), np.max(X[:,features[0]])
    y_min,y_max = np.min(X[:,features[1]]), np.max(X[:,features[1]])
    plt.xlim(x_min-0.2*(x_max-x_min), x_max+0.2*(x_max-x_min))
    plt.ylim(y_min-0.2*(y_max-y_min), y_max+0.2*(y_max-y_min))
    plt.show()