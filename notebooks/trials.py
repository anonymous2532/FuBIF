import numpy as np
from utils.plot import *
import os
import time
import sys
from past_works.EIF_rebootV2 import ExtendedIsolationForest as EIF
import numpy as np
import pandas as pd
from utils.datasets import Dataset
from sklearn.metrics import average_precision_score


start_time = time.time()
dataset = Dataset('wine', path = 'data/real/')
dataset.drop_duplicates()
#dataset.split_dataset(contamination=0)
#dataset.pre_process()

I=EIF(False,n_estimators=300)
I.fit(dataset.X)

print("[AVG PREC] ",average_precision_score(dataset.y,I.predict(dataset.X)))
print("--- %s seconds ---" % (time.time() - start_time))
      
start_time = time.time()
dataset = Dataset('wine', path = 'data/real/')
dataset.drop_duplicates()
#dataset.split_dataset(contamination=0)
#dataset.pre_process()

I=EIF(False,n_estimators=300)
I.fit(dataset.X)

print("[AVG PREC] ",average_precision_score(dataset.y,I.predict(dataset.X)))
print("--- %s seconds ---" % (time.time() - start_time))