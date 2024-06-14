import sys
sys.path.append("")
from tqdm import tqdm
import time
import os 
import copy 

import FuBIF
from FuBIF import ExtendedIsolationForest as IF
from EIF_optimized import ExtendedIsolationForest as EIF
from utils.datasets import Dataset
from split_functions import *

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, average_precision_score, balanced_accuracy_score
import pickle
from argparse import ArgumentParser

# Create argument parser
parser = ArgumentParser()
parser.add_argument('--interpretation_model', type=int, help="number of interpretation model", default=0)

# Parse the command-line arguments
args = parser.parse_args()
function_n = args.interpretation_model

# Access the interpretation model argument
interpretation_model = args.interpretation_model

functions = ["Ellipsoid"]

t = time.localtime()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", t)

n_runs = 10

# path = 'Yourpath/FuBIF/results/numeric/interpretations/'
# file = sorted(os.listdir('Yourpath/FuBIF/results/numeric/interpretations/'))[-1]
path_results = "Yourpath/FuBIF/results/numeric/interpretations/Ellipsoid_2024-06-14_03-05-41.pkl"
print(f"Loading interpretation from: {path_results}")
with open(path_results, 'rb') as f:
    interpretation = pickle.load(f)


    
complete_interpretation_results = {}
for split_function in functions:
    print(f"Processing interpretation of split function: {split_function}")
    interpretation_function = interpretation[split_function]
    values = {}

    for dataset_name in tqdm(os.listdir('data/real/') + os.listdir('data/syn/')):
        dataset_name = dataset_name.split('.')[0]
        print(f"Processing dataset: {dataset_name}")
        values[dataset_name] = {}

        for Scenario_evaluation in [1,2]:
            values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)] = {}
            for plus_evaluation in [True, False]:
                print(f"   Evaluating with [Model] EIF  [Scenario]{Scenario_evaluation} [plus]{plus_evaluation}")
                values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)] = {}
                for Scenario in [1, 2]:
                    values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)] = {}
                    for plus in [True, False]:
                        print(f"    Interpretation of [Model]{split_function}  [Scenario]{Scenario} [plus]{plus}")
                        values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)] = {}
                        interpr = np.mean(interpretation_function[dataset_name]["Scenario" + str(Scenario)]["plus " + str(plus)], axis=0)
                        

                        values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["AUCFS"] = {}
                        values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["direct"] = {}
                        values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["inverse"] = {}

                        values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["direct"]["Avg Prec"] = []
                        values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["direct"]["Prec"] = []
                        values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["direct"]["ROC AUC"] = []

                        values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["inverse"]["Avg Prec"] = []
                        values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["inverse"]["Prec"] = []
                        values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["inverse"]["ROC AUC"] = []

                        try:
                            dataset = Dataset(dataset_name, path='data/real/')
                            print("      Loaded dataset from real data")
                        except:
                            try:
                                dataset = Dataset(dataset_name, path='data/syn/')
                                print("      Loaded dataset from synthetic data")
                            except:
                                print("      Failed to load dataset")
                                continue

                        dataset.drop_duplicates()
                        if Scenario_evaluation == 2:
                            dataset.split_dataset(contamination=0)
                        dataset.pre_process()

                        for feat_number in range(dataset.X_train.shape[1]):
                            print(f"      Processing {dataset.X_train.shape[1]-feat_number}")
                            most_avg_prec_list = []
                            most_prec_list = []
                            most_roc_auc_list = []
                            less_avg_prec_list = []
                            less_prec_list = []
                            less_roc_auc_list = []
                            most_important = np.argsort(interpr)[::-1][:dataset.X_train.shape[1]-feat_number]
                            less_important = np.argsort(interpr)[::-1][feat_number:]

                            dataset_most = copy.deepcopy(dataset)
                            dataset_most.X_train = dataset_most.X_train[:, most_important]
                            dataset_most.X_test = dataset_most.X_test[:, most_important]
                            dataset_most.drop_duplicates()

                            dataset_less = copy.deepcopy(dataset)
                            dataset_less.X_train = dataset_less.X_train[:, less_important]
                            dataset_less.X_test = dataset_less.X_test[:, less_important]
                            dataset_less.drop_duplicates()
                            for runs in tqdm(range(n_runs)):

                                I_most = EIF(plus_evaluation, n_estimators=400)
                                I_most.fit(dataset_most.X_train)
                                y_most_scores = I_most.predict(dataset_most.X_test)

                                I_less = EIF(plus_evaluation, n_estimators=400)
                                I_less.fit(dataset_less.X_train)
                                y_less_scores = I_less.predict(dataset_less.X_test)

                                most_avg_prec = average_precision_score(dataset.y_test, y_most_scores)
                                most_prec = precision_score(dataset.y_test, I_most._predict(dataset_most.X_test, p=dataset_most.perc_outliers))
                                most_roc_auc = roc_auc_score(dataset.y_test, y_most_scores)
                                
                                less_avg_prec = average_precision_score(dataset.y_test, y_less_scores)
                                less_prec = precision_score(dataset.y_test, I_less._predict(dataset_less.X_test, p=dataset_less.perc_outliers))
                                less_roc_auc = roc_auc_score(dataset.y_test, y_less_scores)
                                
                                most_avg_prec_list.append(most_avg_prec)
                                most_prec_list.append(most_prec)
                                most_roc_auc_list.append(most_roc_auc)
                                less_avg_prec_list.append(less_avg_prec)
                                less_prec_list.append(less_prec)
                                less_roc_auc_list.append(less_roc_auc)
                                
                            print(f"        Most {dataset.X_train.shape[1]-feat_number} important features: Avg Prec: {np.mean(most_avg_prec_list)}, Prec: {np.mean(most_prec_list)}, ROC AUC: {np.mean(most_roc_auc_list)}")
                            values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["direct"]["Avg Prec"].append(most_avg_prec_list)
                            values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["direct"]["Prec"].append(most_prec_list)
                            values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["direct"]["ROC AUC"].append(most_roc_auc_list)

                            print(f"        Less {dataset.X_train.shape[1]- feat_number} important features: Avg Prec: {np.mean(less_avg_prec_list)}, Prec: {np.mean(less_prec_list)}, ROC AUC: {np.mean(less_roc_auc_list)}")
                            values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["inverse"]["Avg Prec"].append(less_avg_prec_list)
                            values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["inverse"]["Prec"].append(less_prec_list)
                            values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["inverse"]["ROC AUC"].append(less_roc_auc_list)

                        values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["AUCFS"]["Avg Prec"] = np.mean(np.mean(values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["direct"]["Avg Prec"],axis = 0) - np.mean(values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["inverse"]["Avg Prec"],axis = 0))
                        values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["AUCFS"]["Prec"] = np.mean(np.mean(values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["direct"]["Prec"],axis = 0) - np.mean(values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["inverse"]["Prec"],axis = 0) )
                        values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["AUCFS"]["ROC AUC"] = np.mean(np.mean(values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["direct"]["ROC AUC"],axis = 0) - np.mean(values[dataset_name]["Scenario evaluation " + str(Scenario_evaluation)]["plus " + str(plus_evaluation)]["Scenario" + str(Scenario)]["plus " + str(plus)]["inverse"]["ROC AUC"],axis = 0) )

    complete_interpretation_results[split_function] = values

    if not os.path.exists('results/numeric/AUCFS/'):
        os.makedirs('results/numeric/AUCFS/')
    with open(f'results/numeric/AUCFS/{split_function}_{current_time}.pkl', 'wb') as f:
        pickle.dump(complete_interpretation_results, f)
    print(f"Saved results for split function: {split_function}")

print("Processing complete")