import sys
sys.path.append("")
from tqdm import tqdm
import time
import os 

import FuBIF_optimized as FuBIF
from FuBIF_optimized import ExtendedIsolationForest as EIF
from utils.datasets import Dataset
from split_functions_optimized import *

from pyod.models.iforest import IForest as IF
from pyod.models.auto_encoder import AutoEncoder as AE
from pyod.models.dif import DIF as DIF

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, average_precision_score, balanced_accuracy_score
import pickle

functions = [IF, AE, DIF]

t = time.localtime()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", t)

n_runs = 10

complete_results = {}

for function in functions:
    print(f"Processing split function: {function.__name__}")
    results_values = {}

    for dataset_name in tqdm(os.listdir('data/real/') + os.listdir('data/syn/')):
        dataset_name = dataset_name.split('.')[0]
        print(f"Processing dataset: {dataset_name}")
        results_values[dataset_name] = {}
        for Scenario in [1, 2]:
            print(f"  Scenario: {Scenario}")
            results_values[dataset_name]["Scenario" + str(Scenario)] = {}
            for plus in [True, False]:
                print(f"    Plus: {plus}")
                results_values[dataset_name]["Scenario" + str(Scenario)]["plus " + str(plus)] = {}

                results_values[dataset_name]["Scenario" + str(Scenario)]["plus " + str(plus)]["Avg Prec"] = []
                results_values[dataset_name]["Scenario" + str(Scenario)]["plus " + str(plus)]["Prec"] = []
                results_values[dataset_name]["Scenario" + str(Scenario)]["plus " + str(plus)]["ROC AUC"] = []
                for n in range(n_runs):
                    print(f"      Run: {n+1}/{n_runs}")
                    try:
                        dataset = Dataset(dataset_name, path='data/real/')
                        print("        Loaded from real data")
                    except:
                        try:
                            dataset = Dataset(dataset_name, path='data/syn/')
                            print("        Loaded from synthetic data")
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
                    I.fit(dataset.X_train)
                    print("        Fitted model")
                    y_scores = I.decision_function(dataset.X_test)
                    print("        Predicted scores")

                    avg_prec = average_precision_score(dataset.y_test, y_scores)
                    prec = precision_score(dataset.y_test, I.predict(dataset.X_test))
                    roc_auc = roc_auc_score(dataset.y_test, y_scores)
                    print(f"        Avg Prec: {avg_prec}, Prec: {prec}, ROC AUC: {roc_auc}")

                    results_values[dataset_name]["Scenario" + str(Scenario)]["plus " + str(plus)]["Avg Prec"].append(avg_prec)
                    results_values[dataset_name]["Scenario" + str(Scenario)]["plus " + str(plus)]["Prec"].append(prec)
                    results_values[dataset_name]["Scenario" + str(Scenario)]["plus " + str(plus)]["ROC AUC"].append(roc_auc)

    complete_results[function.__name__] = results_values    
    if not os.path.exists('results/numeric/performances/'):
        os.makedirs('results/numeric/performances/')
    with open(f'results/numeric/performances/{current_time}.pkl', 'wb') as f:
        pickle.dump(complete_results, f)
    print(f"Saved results for split function: {function.__name__}")

print("Processing complete")


