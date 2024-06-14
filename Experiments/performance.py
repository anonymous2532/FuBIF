import sys
sys.path.append("")
from tqdm import tqdm
import time
import os 

import FuBIF
from FuBIF import ExtendedIsolationForest as EIF
from utils.datasets import Dataset
from split_functions import *

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, average_precision_score, balanced_accuracy_score
import pickle

functions = [x for x in splitting_functions_dictionary.values() if x.__name__ == 'EllipsoidSplitting']

t = time.localtime()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", t)

n_runs = 10

complete_interpretations = {}
complete_results = {}

for split_function in functions:
    print(f"Processing split function: {split_function.__name__}")
    FuBIF.splitting_functions = [split_function]
    results_values = {}
    interpretation_values = {}

    for dataset_name in tqdm(os.listdir('data/real/') + os.listdir('data/syn/')):
        dataset_name = dataset_name.split('.')[0]
        print(f"Processing dataset: {dataset_name}")
        results_values[dataset_name] = {}
        interpretation_values[dataset_name] = {}
        for Scenario in [1, 2]:
            print(f"  Scenario: {Scenario}")
            results_values[dataset_name]["Scenario" + str(Scenario)] = {}
            interpretation_values[dataset_name]["Scenario" + str(Scenario)] = {}
            for plus in [True, False]:
                print(f"    Plus: {plus}")
                results_values[dataset_name]["Scenario" + str(Scenario)]["plus " + str(plus)] = {}
                interpretation_values[dataset_name]["Scenario" + str(Scenario)]["plus " + str(plus)] = []

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

                    I = EIF(plus, n_estimators=400)
                    I.fit(dataset.X_train)
                    print("        Fitted EIF model")
                    y_scores = I.predict(dataset.X_test)
                    print("        Predicted scores")

                    interpretation_values[dataset_name]["Scenario" + str(Scenario)]["plus " + str(plus)].append(I.global_importances(dataset.X_test))

                    avg_prec = average_precision_score(dataset.y_test, y_scores)
                    prec = precision_score(dataset.y_test, I._predict(dataset.X_test, p=dataset.perc_outliers))
                    roc_auc = roc_auc_score(dataset.y_test, y_scores)
                    print(f"        Avg Prec: {avg_prec}, Prec: {prec}, ROC AUC: {roc_auc}")

                    results_values[dataset_name]["Scenario" + str(Scenario)]["plus " + str(plus)]["Avg Prec"].append(avg_prec)
                    results_values[dataset_name]["Scenario" + str(Scenario)]["plus " + str(plus)]["Prec"].append(prec)
                    results_values[dataset_name]["Scenario" + str(Scenario)]["plus " + str(plus)]["ROC AUC"].append(roc_auc)

    complete_results[split_function.__name__] = results_values
    complete_interpretations[split_function.__name__] = interpretation_values
    
    if not os.path.exists('results/numeric/performances/'):
        os.makedirs('results/numeric/performances/')
    with open(f'results/numeric/performances/{current_time}.pkl', 'wb') as f:
        pickle.dump(complete_results, f)
    if not os.path.exists('results/numeric/interpretations/'):
        os.makedirs('results/numeric/interpretations/')
    with open(f'results/numeric/interpretations/{current_time}.pkl', 'wb') as f:
        pickle.dump(complete_interpretations, f)
    print(f"Saved results for split function: {split_function.__name__}")

print("Processing complete")