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

SEED = 42
N_sample = 250
tune = 1000
chains = 4
draws = 1000
notice = 'without intercept'
model_path = "./simu_" + notice + '_' + str(N_sample) + '_' + str(draws) + '_' + str(tune) + '_' + str(chains) + '/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
np.random.seed(SEED)
dataset = SimulatedData(N_sample)
dataset.explore_dataset()
x = dataset['x'].values
y = dataset['y'].values
nbr_classes = dataset.nbr_classes

with pm.Model() as ordered_multinomial:

    cutpoints = pm.Normal(
        "cutpoints",
        0.0,
        1.5,
        transform=pm.distributions.transforms.ordered,
        shape=nbr_classes,
        testval=np.arange(nbr_classes) - (nbr_classes-1)/2,
    )
    #  a = pm.Normal("intercept", mu=0, sigma=1.0)  # intercepts
    b = pm.Normal("age", mu=0, sigma=1.5)
    c = pm.Normal("gender", mu=0, sigma=1.5)
    d = pm.Normal("smoking", mu=0, sigma=1.5)
    e = pm.Normal("fever", mu=0, sigma=1.5)
    f = pm.Normal("vomiting", mu=0, sigma=1.5)
    phi = b * x[:, 0] + c * x[:, 1] + d * x[:, 2] + e * x[:, 3] + f * x[:, 4]
    outcome = pm.OrderedLogistic("ordered_outcome", eta=phi, cutpoints=cutpoints, observed=y)
    trace_ordered_multinomial = pm.sample(draws=draws,
                           tune=tune,
                           chains=chains,
                           cores=1,
                           init='auto', progressbar=True)

    print(ordered_multinomial.basic_RVs)
    pm.plot_trace(trace_ordered_multinomial)
    plt.savefig(model_path + 'trace')
    pm.summary(trace_ordered_multinomial).to_csv(model_path + 'trace.csv')
    print(pm.summary(trace_ordered_multinomial))
    az.plot_pair(trace_ordered_multinomial)
    plt.savefig(model_path + 'plot_pair')
    pm.energyplot(trace_ordered_multinomial)
    plt.savefig(model_path + 'energy')
    az.plot_forest(trace_ordered_multinomial)
    plt.savefig(model_path + 'forest')
    pm.plot_posterior(trace_ordered_multinomial)
    plt.savefig(model_path + 'posterior')





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