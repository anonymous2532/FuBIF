import numpy as np
from utils.plot import *
import os
import time
import sys
from FuBIF import ExtendedIsolationForest as EIF
import numpy as np
import pandas as pd
from utils.datasets import Dataset
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from split_functions import Hyperplane

start_time = time.time()
dataset = Dataset('cardio', path = 'data/real/')
dataset.drop_duplicates()
dataset.split_dataset(contamination=0)
dataset.pre_process()


I=EIF(True,n_estimators=400)
I.fit(dataset.X_train)

print("--- %s [FIT] seconds ---" % (time.time() - start_time))

start_time = time.time()
print("[AVG PREC] ",average_precision_score(dataset.y,I.predict(dataset.X_test)))
print("--- %s [PREDICT] seconds ---" % (time.time() - start_time))