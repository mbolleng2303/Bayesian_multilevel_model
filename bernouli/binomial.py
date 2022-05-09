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
N_sample = 1000
tune = 1000
chains = 4
draws = 1000
notice = 'glm_binomial'
model_path = "./simu_" + notice + '_' + str(N_sample) + '_' + str(draws) + '_' + str(tune) + '_' + str(chains) + '/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
np.random.seed(SEED)
dataset = SimulatedData(N_sample, outcome='ICU')
dataset.explore_dataset()
x = dataset['x'].values
y = dataset['y'].values
nbr_classes = dataset.nbr_classes
labels = dataset['x'].columns
with pm.Model() as binomial_model:
    pm.glm.GLM(x=x, labels=labels,
               y=y, family=pm.glm.families.Binomial())

    trace_binomial = pm.sample(draws=draws,
                               tune=tune,
                               chains=chains,
                               cores=1,
                               init='auto', progressbar=True)

    print(binomial_model.basic_RVs)
    pm.plot_trace(trace_binomial)
    plt.savefig(model_path + 'trace')
    pm.summary(trace_binomial).to_csv(model_path + 'trace.csv')
    print(pm.summary(trace_binomial))
    az.plot_pair(trace_binomial)
    plt.savefig(model_path + 'plot_pair')
    pm.energyplot(trace_binomial)
    plt.savefig(model_path + 'energy')
    az.plot_forest(trace_binomial)
    plt.savefig(model_path + 'forest')
    pm.plot_posterior(trace_binomial)
    plt.savefig(model_path + 'posterior')
