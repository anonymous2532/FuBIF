from typing import Type,Union

import sys
sys.path.append('../')

import numpy as np
import numpy.typing as npt
from tqdm import tqdm,trange
import copy

import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, average_precision_score, balanced_accuracy_score

import pickle
import time
import pandas as pd

from FuBIF import *
from split_functions import *
from utils.datasets import Dataset

import os
cwd = os.getcwd()

filename = cwd + "/utils_reboot/time_scaling_test_dei_new.pickle"


if not os.path.exists(filename):
    dict_time = {"fit":{"EIF+":{},"IF":{},"DIF":{},"EIF":{},"sklearn_IF":{}}, 
            "predict":{"EIF+":{},"IF":{},"DIF":{},"EIF":{},"sklearn_IF":{}},
            "importances":{"EXIFFI+":{},"EXIFFI":{},"DIFFI":{},"RandomForest":{}}}
    
    with open(filename, "wb") as file:
        pickle.dump(dict_time, file)
               
with open(filename, "rb") as file:
    dict_time = pickle.load(file)

    

def compute_global_importances( I: Type[SplittingFunction],
                                dataset: Type[Dataset],
                                p = 0.1,
                                fit_model = True,
                                randomforest = False) -> np.array: 
    
    """
    Compute the global feature importances for an interpration model on a specific dataset.

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Type[Dataset]): Input dataset.
        p (float): The percentage of outliers in the dataset (i.e. contamination factor). Defaults to 0.1.
        interpretation (str): Name of the interpretation method to be used. Defaults to "EXIFFI+".
        fit_model (bool): Whether to fit the model on the dataset. Defaults to True.

    Returns:
        The global feature importance vector.

    """
    if not randomforest:
        if fit_model:
            I.fit(dataset.X_train)        
        fi=I.global_importances(dataset.X_test,p)
    if randomforest:
        rf = RandomForestRegressor()
        rf.fit(dataset.X_test, I.predict(dataset.X_test))
        fi = rf.feature_importances_
    return fi

def fit_predict_experiment(I: Type[ExtendedIsolationForest],
                            dataset: Type[Dataset],
                            n_runs:int = 40,
                            model='EIF+') -> tuple[float,float]:
    
    """
    Fit and predict the model on the dataset for a number of runs and keep track of the fit and predict times.

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Type[Dataset]): Input dataset.
        n_runs (int): The number of runs. Defaults to 40.
        model (str): The name of the model. Defaults to 'EIF+'.

    Returns:
        The average fit and predict time.
    """

    fit_times = []
    predict_times = []
    
    for i in trange(n_runs):
        start_time = time.time()
        I.fit(dataset.X_train)
        fit_time = time.time() - start_time
        if i>3:  
            fit_times.append(fit_time)
            dict_time["fit"][I.name].setdefault(dataset.name, []).append(fit_time) 
        
        start_time = time.time()
        if model in ['EIF','EIF+']:
            _=I._predict(dataset.X_test,p=dataset.perc_outliers)
            predict_time = time.time() - start_time
        elif model in ['sklearn_IF','DIF','AnomalyAutoencoder']:
            _=I.predict(dataset.X_test)
            predict_time = time.time() - start_time

        if i>3:
            predict_times.append(predict_time)
            dict_time["predict"][I.name].setdefault(dataset.name, []).append(predict_time)

    with open(filename, "wb") as file:
        pickle.dump(dict_time, file)

    return np.mean(fit_times), np.mean(predict_times)
                        
def experiment_global_importances(I:Type[ExtendedIsolationForest],
                               dataset:Type[Dataset],
                               n_runs:int=10, 
                               p:float=0.1,
                               model:str="EIF+",
                               interpretation:str="EXIFFI+"
                               ) -> tuple[np.array,dict,str,str]:
    
    """
    Compute the global feature importances for an interpration model on a specific dataset for a number of runs.

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Type[Dataset]): Input dataset.
        n_runs (int): The number of runs. Defaults to 10.
        p (float): The percentage of outliers in the dataset (i.e. contamination factor). Defaults to 0.1.
        model (str): The name of the model. Defaults to 'EIF+'.
        interpretation (str): Name of the interpretation method to be used. Defaults to "EXIFFI+".
    
    Returns:
        The global feature importances vectors for the different runs and the average importances times.
    """

    fi=np.zeros(shape=(n_runs,dataset.X.shape[1]))
    imp_times=[]
    for i in tqdm(range(n_runs)):
        start_time = time.time()
        fi[i,:]=compute_global_importances(I,
                        dataset,
                        p = p,
                        interpretation=interpretation,
                        model = model)
        gfi_time = time.time() - start_time
        if i>3:
            imp_times.append(gfi_time)
            dict_time["importances"][interpretation].setdefault(dataset.name, []).append(gfi_time)
            #print(f'Added time {str(gfi_time)} to time dict')
    
    with open(filename, "wb") as file:
        pickle.dump(dict_time, file)
    return fi,np.mean(imp_times)

def compute_plt_data(imp_path:str) -> dict:

    """
    Compute statistics on the global feature importances obtained from experiment_global_importances. These will then be used in the score_plot method. 

    Args:
        imp_path (str): The path to the importances file.
    
    Returns:
        The dictionary containing the mean importances, the feature order, and the standard deviation of the importances.
    """

    try:
        fi = np.load(imp_path)['element']
    except:
        print("Error: importances file should be npz")
    # Handle the case in which there are some np.nan in the fi array
    if np.isnan(fi).any():
        #Substitute the np.nan values with 0  
        #fi=np.nan_to_num(fi,nan=0)
        mean_imp = np.nanmean(fi,axis=0)
        std_imp = np.nanstd(fi,axis=0)
    else:
        mean_imp = np.mean(fi,axis=0)
        std_imp = np.std(fi,axis=0)
    
    feat_ordered = mean_imp.argsort()
    mean_ordered = mean_imp[feat_ordered]
    std_ordered = std_imp[feat_ordered]

    plt_data={'Importances': mean_ordered,
                'feat_order': feat_ordered,
                'std': std_ordered}
    return plt_data
    

def feature_selection(I: Type[ExtendedIsolationForest],
                      dataset: Type[Dataset],
                      importances_indexes: npt.NDArray,
                      n_runs: int = 10, 
                      inverse: bool = True,
                      random: bool = False,
                      scenario:int=2
                      ) -> np.array:
        
        """
        Perform feature selection on the dataset by dropping features in order of importance.

        Args:
            I (Type[ExtendedIsolationForest]): The AD model.
            dataset (Type[Dataset]): Input dataset.
            importances_indexes (npt.NDArray): The indexes of the features in the dataset.
            n_runs (int): The number of runs. Defaults to 10.
            inverse (bool): Whether to drop the features in decreasing order of importance. Defaults to True.
            random (bool): Whether to drop the features in random order. Defaults to False.
            scenario (int): The scenario of the experiment. Defaults to 2.
        
        Returns:
            The average precision scores for the different runs.
        """

        dataset_shrinking = copy.deepcopy(dataset)
        d = dataset.X.shape[1]
        precisions = np.zeros(shape=(len(importances_indexes),n_runs))
        for number_of_features_dropped in tqdm(range(len(importances_indexes))):
            runs = np.zeros(n_runs)
            for run in range(n_runs):
                if random:
                    importances_indexes = np.random.choice(importances_indexes, len(importances_indexes), replace=False)
                dataset_shrinking.X = dataset.X_test[:,importances_indexes[:d-number_of_features_dropped]] if not inverse else dataset.X_test[:,importances_indexes[number_of_features_dropped:]]
                dataset_shrinking.y = dataset.y
                dataset_shrinking.drop_duplicates()
                
                if scenario==2:
                    dataset_shrinking.split_dataset(1-dataset_shrinking.perc_outliers,0)
                    dataset_shrinking.initialize_test()
                else:
                    dataset_shrinking.initialize_train()
                    dataset_shrinking.initialize_test()

                try:
                    if dataset.X.shape[1] == dataset_shrinking.X.shape[1]:
                        
                        start_time = time.time()
                        I.fit(dataset_shrinking.X_train)
                        fit_time = time.time() - start_time
                        
                        if run >3:
                            dict_time["fit"][I.name].setdefault(dataset.name, []).append(fit_time)
                        start_time = time.time()
                        score = I.predict(dataset_shrinking.X_test)
                        predict_time = time.time() - start_time
                        
                        if run >3:                        
                            dict_time["predict"][I.name].setdefault(dataset.name, []).append(predict_time)
                    else:
                        I.fit(dataset_shrinking.X_train)
                        score = I.predict(dataset_shrinking.X_test)
                    avg_prec = sklearn.metrics.average_precision_score(dataset_shrinking.y,score)
                    # import ipdb;
                    # ipdb.set_trace()
                    runs[run] = avg_prec
                except:
                    runs[run] = np.nan

            precisions[number_of_features_dropped] = runs

        with open(filename, "wb") as file:
            pickle.dump(dict_time, file)
        return precisions
    

def contamination_in_training_precision_evaluation(I: Type[ExtendedIsolationForest],
                                                   dataset: Type[Dataset],
                                                   n_runs: int = 10,
                                                   train_size = 0.8,
                                                   contamination_values: npt.NDArray = np.linspace(0.0,0.1,10),
                                                   compute_GFI:bool=False,
                                                   interpretation:str="EXIFFI+",
                                                   pre_process:bool=True, # in the synthetic datasets the dataset should not be pre processed
                                                   ) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
    
    """
    Evaluate the average precision of the model on the dataset for different contamination values in the training set. 
    The precision values will then be used in the `plot_precision_over_contamination` method

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Type[Dataset]): Input dataset.
        n_runs (int): The number of runs. Defaults to 10.
        train_size (float): The size of the training set. Defaults to 0.8.
        contamination_values (npt.NDArray): The contamination values. Defaults to `np.linspace(0.0,0.1,10)`.
        compute_GFI (bool): Whether to compute the global feature importances. Defaults to False.
        interpretation (str): Name of the interpretation method to be used. Defaults to "EXIFFI+".
        pre_process (bool): Whether to pre process the dataset. Defaults to True.

    Returns:
        The average precision scores and the global feature importances if `compute_GFI` is True, 
        otherwise just the average precision scores are returned. 
    """

    precisions = np.zeros(shape=(len(contamination_values),n_runs))
    if compute_GFI:
        importances = np.zeros(shape=(len(contamination_values),n_runs,len(contamination_values),dataset.X.shape[1]))
    for i,contamination in tqdm(enumerate(contamination_values)):
        for j in range(n_runs):
            dataset.split_dataset(train_size,contamination)
            dataset.initialize_test()

            if pre_process:
                dataset.pre_process()
            
            start_time = time.time()
            I.fit(dataset.X_train)
            fit_time = time.time() - start_time
            
            if j>3:
                try:
                    dict_time["fit"][I.name].setdefault(dataset.name, []).append(fit_time)
                except:
                    print('Model not recognized: creating a new key in the dict_time for the new model')
                    dict_time["fit"].setdefault(I.name, {}).setdefault(dataset.name, []).append(fit_time)
            
            if compute_GFI:
                for k,c in enumerate(contamination_values):
                    start_time = time.time()
                    importances[i,j,k,:] = compute_global_importances(I,
                                                                    dataset,
                                                                    p=c,
                                                                    interpretation=interpretation,
                                                                    fit_model=False)
                    gfi_time = time.time() - start_time
                    if k>3: 
                        dict_time["importances"][interpretation].setdefault(dataset.name, []).append(gfi_time)
                    
            start_time = time.time()
            score = I.predict(dataset.X_test)
            predict_time = time.time() - start_time
            if j>3:
                try:
                    dict_time["predict"][I.name].setdefault(dataset.name, []).append(predict_time)
                except:
                    print('Model not recognized: creating a new key in the dict_time for the new model')
                    dict_time["predict"].setdefault(I.name, {}).setdefault(dataset.name, []).append(predict_time)
            
            avg_prec = sklearn.metrics.average_precision_score(dataset.y_test,score)
            #import ipdb; ipdb.set_trace()
            precisions[i,j] = avg_prec
    
    with open(filename, "wb") as file:
        pickle.dump(dict_time, file)
    if compute_GFI:
        return precisions,importances
    return precisions

def performance(y_pred:np.array,
                y_true:np.array,
                score:np.array,
                I:Type[ExtendedIsolationForest],
                model_name:str,
                dataset:Type[Dataset],
                contamination:float=0.1,
                train_size:float=0.8,
                scenario:int=2,
                n_runs:int=10,
                filename:str="",
                path:str=os.getcwd(),
                save:bool=True
                ) -> tuple[pd.DataFrame,str]: 
    
    """
    Compute the performance metrics of the model on the dataset.

    Args:
        y_pred (np.array): The predicted labels.
        y_true (np.array): The true labels.
        score (np.array): The Anomaly Scores.
        I (Type[ExtendedIsolationForest]): The AD model.
        model_name (str): The name of the model.
        dataset (Type[Dataset]): Input dataset.
        contamination (float): The contamination factor. Defaults to 0.1.
        train_size (float): The size of the training set. Defaults to 0.8.
        scenario (int): The scenario of the experiment. Defaults to 2.
        n_runs (int): The number of runs. Defaults to 10.
        filename (str): The filename. Defaults to "".
        path (str): The path to the experiments folder. Defaults to os.getcwd().
        save (bool): Whether to save the results. Defaults to True.

    Returns:
        The performance metrics and the path to the results.
    """
    
    # In path insert the local path up to the experiments folder:
    # For Davide → /home/davidefrizzo/Desktop/PHD/ExIFFI/experiments
    # For Alessio → /Users/alessio/Documents/ExIFFI

    y_pred=y_pred.astype(int)
    y_true=y_true.astype(int)

    if dataset.X.shape[0]>7500:
        dataset.downsample(max_samples=7500)

    precisions=[]
    for i in trange(n_runs):
        I.fit(dataset.X_train)
        if model_name in ['DIF','AnomalyAutoencoder']:
            score = I.decision_function(dataset.X_test)
        else:
            score = I.predict(dataset.X_test)
        precisions.append(average_precision_score(y_true, score))
    
    df=pd.DataFrame({
        "Model": model_name,
        "Dataset": dataset.name,
        "Contamination": contamination,
        "Train Size": train_size,
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "f1 score": f1_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Average Precision": np.mean(precisions),
        "ROC AUC Score": roc_auc_score(y_true, y_pred)
    }, index=[pd.Timestamp.now()])

    path=path + f"/experiments/results/{dataset.name}/experiments/metrics/{model_name}/" + f"scenario_{str(scenario)}/"

    if not os.path.exists(path):
        os.makedirs(path)
    
    filename=f"perf_{dataset.name}_{model_name}_{scenario}"

    if save:
        save_element(df, path, filename)
    
    return df,path

def ablation_EIF_plus(I:Type[ExtendedIsolationForest], 
                      dataset:Type[Dataset], 
                      eta_list:list[float], 
                      nruns:int=10) -> list[np.array]:

    """
    Compute the average precision scores for different values of the eta parameter in the EIF+ model.

    Args:
        I (Type[ExtendedIsolationForest]): The AD model.
        dataset (Type[Dataset]): Input dataset.
        eta_list (list): The list of eta values.
        nruns (int): The number of runs. Defaults to 10.

    Returns:
        The average precision scores.
    """

    precisions = []
    for eta in tqdm(eta_list):
        precision = []
        for run in range(nruns):
            I.eta = eta
            I.fit(dataset.X_train)
            score = I.predict(dataset.X_test)
            precision.append(average_precision_score(dataset.y_test, score))
        precisions.append(precision)
    return precisions
        
        
    


