import warnings
import tqdm as tqdm
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score,
                             precision_recall_curve)
import pickle
import pymc3 as pm
import theano.tensor as tt
from mlxtend.plotting import plot_confusion_matrix
import theano
from pymc3.variational.callbacks import CheckParametersConvergence
import statsmodels.formula.api as smf
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from IPython.display import HTML
from Dataset import SimulatedData
import time
from utils import make_plot

SEED = 42
N_sample = 100
tune = 100
chains = 1
draws = 100
notice = 'without intercept'
model_path = "./simu_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y') + notice + '_' + str(N_sample) + '_' + str(draws) + '_' + str(tune) + '_' + str(chains) + '/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
np.random.seed(SEED)
dataset = SimulatedData(N_sample)
dataset.explore_dataset()
x = dataset['x'].values
y = dataset['y'].values
true_coeff = dataset.get_true_coeff()
nbr_classes = dataset.nbr_classes

with pm.Model() as ordered_multinomial:

    cutpoints = pm.Normal(
        "cutpoints",
        0.0,
        100,
        transform=pm.distributions.transforms.ordered,
        shape=nbr_classes-1,
        testval=np.arange(nbr_classes-1) - (nbr_classes-1)/2,
    )
    #  a = pm.Normal("intercept", mu=0, sigma=1.0)  # intercepts
    b = pm.Normal("age", mu=0, sigma=50)
    c = pm.Normal("gender", mu=0, sigma=50)
    d = pm.Normal("smoking", mu=0, sigma=50)
    e = pm.Normal("fever", mu=0, sigma=50)
    f = pm.Normal("vomiting", mu=0, sigma=50)
    phi = b * x[:, 0] + c * x[:, 1] + d * x[:, 2] + e * x[:, 3] + f * x[:, 4]
    outcome = pm.OrderedLogistic("ordered_outcome", eta=phi, cutpoints=cutpoints, observed=y)
    trace_ordered_multinomial = pm.sample(draws=draws,
                           tune=tune,
                           chains=chains,
                           cores=1,
                           init='auto', progressbar=True)

    with open(model_path + 'model.pkl', 'wb') as buff:
        pickle.dump({'model': ordered_multinomial, 'trace': trace_ordered_multinomial}, buff)
    with open(model_path + 'model.pkl', 'rb') as buff:
        data0 = pickle.load(buff)
    model, trace = data0['model'], data0['trace']
    make_plot(model, trace, true_coeff,
              model_path)







'''X_shared.set_value(X_train)
ppc_t = pm.sample_posterior_predictive(trace_ordered_multinomial,
                    model=ordered_multinomial,
                    samples=1000)

y_score = np.round(np.mean(ppc_t['outcome'], axis=0))
acc = accuracy_score(y_train, y_score)
print("AUC for trainset = ", acc)

X_shared.set_value(X_test)
ppc = pm.sample_posterior_predictive(trace_ordered_multinomial,
                    model=ordered_multinomial,
                    samples=1000)

# pm.plot_trace(trace_multinomial)
# plt.show()
y_score = np.round(np.mean(ppc_t['outcome'], axis=0))
acc = accuracy_score(y_train, y_score)
print("acc for testset = ", acc)
'''# az.summary(trace_multinomial)