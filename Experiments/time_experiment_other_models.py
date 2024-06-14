import sys
sys.path.append("")
from tqdm import tqdm
import time
import os 

import FuBIF_optimized as FuBIF
from FuBIF_optimized import ExtendedIsolationForest as EIF
from utils.datasets import Dataset
from split_functions_optimized import *

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, average_precision_score, balanced_accuracy_score
import pickle

from pyod.models.iforest import IForest as IF
from pyod.models.auto_encoder import AutoEncoder as AE
from pyod.models.dif import DIF as DIF

#functions = ["NN"] #["Hyperplane","Hypersphere", "SingleDimension", "Hyperbolic", "Ellipsoid", "Paraboloid", "Quadric"]
#functions = [IF, AE, DIF]
functions = [DIF]
print([i.__name__ for i in functions])

t = time.localtime()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", t)

n_runs = 10

complete_time={}


for function in functions:
    print(f"Processing split function: {function.__name__}")
    results_values = {}

    for dataset_file in tqdm(os.listdir('data/time_test/')):
        dataset_n = dataset_file.split('.')[0].split('_')[1]
        dataset_d = dataset_file.split('.')[0].split('_')[2]
        print(f"Processing dataset: {dataset_n}, {dataset_d}")
        dataset_name = dataset_n+"_"+dataset_d
        results_values[dataset_name] = {}
        
        Scenario = 2
        plus = True


        results_values[dataset_name]["Avg Prec"] = []
        results_values[dataset_name]["Prec"] = []
        results_values[dataset_name]["ROC AUC"] = []
        results_values[dataset_name]["interpretation"] = []
        results_values[dataset_name]["Time_fit"] = []
        results_values[dataset_name]["Time_predict"] = []
        results_values[dataset_name]["Time_interpretations"] = []
        for n in tqdm(range(n_runs)):
            try:
                dataset = Dataset(dataset_file.split('.')[0], path='data/time_test/')
            except:
                print("        Failed to load dataset")
                continue

            dataset.drop_duplicates()
            print("        Dropped duplicates")
            if Scenario == 2:
                dataset.split_dataset(contamination=0)
                print("        Split dataset for Scenario 2")
            dataset.pre_process()
            print("        Preprocessed dataset")

            if function.__name__ == "DIF":
                I = function(max_samples=min(256,dataset.X_train.shape[0]), batch_size = 100, representation_dim = 5, hidden_activation = "relu", n_ensemble=10, n_estimators=40, contamination=min(0.5,sum(dataset.y_test)/len(dataset.y_test)))
            elif function.__name__ == "AutoEncoder":
                I = function(hidden_neurons=[dataset.X_train.shape[1],dataset.X_train.shape[1]*3,dataset.X_train.shape[1]*3,dataset.X_train.shape[1]],contamination=min(0.5,sum(dataset.y_test)/len(dataset.y_test)))
            elif function.__name__ == "IForest":
                I = function(n_estimators=400,contamination=min(0.5,sum(dataset.y_test)/len(dataset.y_test)))
            time_start_fit = time.time()
            I.fit(dataset.X_train)
            time_end_fit = time.time()
            print("        Fitted EIF model")
            time_start_predict = time.time()
            y_scores = I.decision_function(dataset.X_test)
            time_end_predict = time.time()
            print("        Predicted scores")
            time_start_interpretations = time.time()
            interpretations = np.zeros(dataset.X_test.shape)
            time_end_interpretations = time.time()
            print("        Interpreted scores")
            
            
            


            avg_prec = average_precision_score(dataset.y_test, y_scores)
            prec = precision_score(dataset.y_test, I.predict(dataset.X_test))
            roc_auc = roc_auc_score(dataset.y_test, y_scores)
            print(f"        Avg Prec: {avg_prec}, Prec: {prec}, ROC AUC: {roc_auc}")
            
            if n>=1:
                results_values[dataset_name]["Avg Prec"].append(avg_prec)
                results_values[dataset_name]["Prec"].append(prec)
                results_values[dataset_name]["ROC AUC"].append(roc_auc)
                results_values[dataset_name]["interpretation"].append(interpretations)
                results_values[dataset_name]["Time_fit"].append(time_end_fit-time_start_fit)
                results_values[dataset_name]["Time_predict"].append(time_end_predict-time_start_predict)
                results_values[dataset_name]["Time_interpretations"].append(np.nan)

        complete_time[function.__name__] = results_values
        
        if not os.path.exists('results/numeric/time/'):
            os.makedirs('results/numeric/time/')
        with open(f'results/numeric/time/{current_time}.pkl', 'wb') as f:
            pickle.dump(complete_time, f)
        print(f"Saved results for split function: {function.__name__}")

print("Processing complete")