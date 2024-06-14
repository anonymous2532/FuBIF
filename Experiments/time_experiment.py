

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
import argparse



FuBIF, FuBIF.ExtendedIsolationForest

# Create the parser
parser = argparse.ArgumentParser(description='Process a string argument.')

# Add the argument
parser.add_argument('--model', type=str, help='The model argument.')

# Parse the arguments
args = parser.parse_args()

# Access the string argument
split_function = args.model
name = split_function

FuBIF.name = name
# Define the splitting function and its derivati
FuBIF.treshold = treshold_calculation
if name == "Hyperplane":
    FuBIF.splitting_parameters_function = hyperplane_generate_parameters
    FuBIF.splitting_function = hyperplane_function
    FuBIF.splitting_derivative = hyperplane_Jacobian
elif name == "Hypersphere":
    FuBIF.splitting_parameters_function = hypersphere_generate_parameters
    FuBIF.splitting_function = hypersphere_function
    FuBIF.splitting_derivative = hypersphere_Jacobian
elif name == "SingleDimension":
    FuBIF.splitting_parameters_function = onedim_generate_parameters
    FuBIF.splitting_function = onedim_function
    FuBIF.splitting_derivative = onedim_Jacobian
elif name == "X2MinusSinX1":
    FuBIF.splitting_parameters_function = X2MinusSinX1_generate_parameters
    FuBIF.splitting_function = X2MinusSinX1_function
    FuBIF.splitting_derivative = X2MinusSinX1_Jacobian
elif name == "Hyperbolic":
    FuBIF.splitting_parameters_function = Hyperbolic_generate_parameters
    FuBIF.splitting_function = Hyperbolic_function
    FuBIF.splitting_derivative = Hyperbolic_Jacobian
elif name == "Ellipsoid":
    FuBIF.splitting_parameters_function = Ellipsoid_generate_parameters
    FuBIF.splitting_function = Ellipsoid_function
    FuBIF.splitting_derivative = Ellipsoid_Jacobian
elif name == "Paraboloid":
    FuBIF.splitting_parameters_function = Paraboloid_generate_parameters
    FuBIF.splitting_function = Paraboloid_function
    FuBIF.splitting_derivative = Paraboloid_Jacobian
elif name == "Quadric":
    FuBIF.splitting_parameters_function = Conic_generate_parameters
    FuBIF.splitting_function = Conic_function
    FuBIF.splitting_derivative = Conic_Jacobian
elif name == "NN":
    FuBIF.splitting_parameters_function = NN_generate_parameters
    FuBIF.splitting_function = NN_function
    FuBIF.splitting_derivative = NN_Jacobian
    
EIF = FuBIF.ExtendedIsolationForest

t = time.localtime()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", t)

n_runs = 10

with open("Yourpath/FuBIF/results/numeric/time/2024-06-11_18-10-45.pkl", "rb") as f:
    complete_time = pickle.load(f)




print(f"Processing split function: {split_function}")
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

        I = EIF(plus, n_estimators=400)
        time_start_fit = time.time()
        I.fit(dataset.X_train)
        time_end_fit = time.time()
        print("        Fitted EIF model")
        time_start_predict = time.time()
        y_scores = I.predict(dataset.X_test)
        time_end_predict = time.time()
        print("        Predicted scores")
        time_start_interpretations = time.time()
        interpretations = I.global_importances(dataset.X_test)
        time_end_interpretations = time.time()
        print("        Interpreted scores")
        
        
        


        avg_prec = average_precision_score(dataset.y_test, y_scores)
        prec = precision_score(dataset.y_test, I._predict(dataset.X_test, p=dataset.perc_outliers))
        roc_auc = roc_auc_score(dataset.y_test, y_scores)
        print(f"        Avg Prec: {avg_prec}, Prec: {prec}, ROC AUC: {roc_auc}")
        
        if n>=1:
            results_values[dataset_name]["Avg Prec"].append(avg_prec)
            results_values[dataset_name]["Prec"].append(prec)
            results_values[dataset_name]["ROC AUC"].append(roc_auc)
            results_values[dataset_name]["interpretation"].append(interpretations)
            results_values[dataset_name]["Time_fit"].append(time_end_fit-time_start_fit)
            results_values[dataset_name]["Time_predict"].append(time_end_predict-time_start_predict)
            results_values[dataset_name]["Time_interpretations"].append(time_end_interpretations-time_start_interpretations)

    complete_time[split_function] = results_values
    
    if not os.path.exists('results/numeric/time/'):
        os.makedirs('results/numeric/time/')
    with open(f'results/numeric/time/{split_function}_{current_time}.pkl', 'wb') as f:
        pickle.dump(complete_time, f)
    print(f"Saved results for split function: {split_function}")

print("Processing complete")