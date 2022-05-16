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
N_sample = 300
tune = 300
chains = 1
draws = 300
notice = 'missing_gender'
model_path = "./result/ordered_" + time.strftime('%Hh%Mm_%m_%d_%y') + '_' + notice + '_' + str(N_sample) + '_' + str(draws) + '_' + str(tune) + '_' + str(chains) + '/'

if not os.path.exists(model_path):
    os.makedirs(model_path)
np.random.seed(SEED)
dataset = SimulatedData(N_sample)
dataset.explore_dataset()
x = dataset['x'].values
y = dataset['y'].values
true_coeff = dataset.get_true_coeff()
prob = dataset.prob[0, 0]
nbr_classes = dataset.nbr_classes
# mask data
missing = 0.05
idx_missing = np.random.choice(x[:, 1].shape[0], int(N_sample*missing), replace=False)
previous = x[idx_missing, 1]
x[idx_missing, 1] = -1
idx_missing = np.where(x[:, 1] == -1)
idx_not_missing = np.where(x[:, 1] != -1)

with pm.Model() as ordered_multinomial:
    w = pm.Dirichlet('w', a=np.array([1, 1]))
    '''cutpoints = pm.Normal(
        "cutpoints",
        0.0,
        100,
        transform=pm.distributions.transforms.ordered,
        shape=nbr_classes - 1,
        testval=np.array(true_coeff[x.shape[1]:x.shape[1]+nbr_classes]),
    )
    #  a = pm.Normal("intercept", mu=true_coeff[0], sigma=5)  # intercepts
    b = pm.Normal("age", mu=true_coeff[0], sigma=5)
    c = pm.Normal("gender", mu=true_coeff[1], sigma=5)
    d = pm.Normal("smoking", mu=true_coeff[2], sigma=5)
    e = pm.Normal("fever", mu=true_coeff[3], sigma=5)
    f = pm.Normal("vomiting", mu=true_coeff[4], sigma=5)'''
    cutpoints = pm.Normal(
        "cutpoints",
        0.0,
        100,
        transform=pm.distributions.transforms.ordered,
        shape=nbr_classes-1,
        testval=np.arange(nbr_classes-1) - (nbr_classes-1)/2,
    )
    #  a = pm.Normal("intercept", mu=0, sigma=100)  # intercepts
    b = pm.Normal("age", mu=0, sigma=10)
    c = pm.Normal("gender", mu=0, sigma=50)
    d = pm.Normal("smoking", mu=0, sigma=50)
    e = pm.Normal("fever", mu=0, sigma=50)
    f = pm.Normal("vomiting", mu=0, sigma=100)
    phi = b * x[idx_not_missing, 0] + c * x[idx_not_missing, 1] + d * x[idx_not_missing, 2] + e * x[
        idx_not_missing, 3] + f * x[idx_not_missing, 4]
    phi_male = b * x[idx_missing, 0] + c * 1 + d * x[idx_missing, 2] + e * x[idx_missing, 3] + f * x[idx_missing, 4]
    phi_female = b * x[idx_missing, 0] + c * 0 + d * x[idx_missing, 2] + e * x[idx_missing, 3] + f * x[
        idx_missing, 4]
    pm.OrderedLogistic("ordered_outcome", eta=phi, cutpoints=cutpoints, observed=y[idx_not_missing])
    logp1 = pm.OrderedLogistic.dist(eta=phi_male, cutpoints=cutpoints)
    logp2 = pm.OrderedLogistic.dist(eta=phi_female, cutpoints=cutpoints)
    pm.Mixture('ordered_missing', w=w, comp_dists=[logp1, logp2], observed=y[idx_missing])
    trace_ordered_multinomial = pm.sample(draws=draws,
                           tune=tune,
                           chains=chains,
                           cores=1,
                           init='auto', progressbar=True)

    '''with open(model_path + 'model.pkl', 'wb') as buff:
        pickle.dump({'model': ordered_multinomial, 'trace': trace_ordered_multinomial}, buff)
    with open(model_path + 'model.pkl', 'rb') as buff:
        data0 = pickle.load(buff)
    model, trace = data0['model'], data0['trace']'''
    model, trace = ordered_multinomial, trace_ordered_multinomial
    make_plot(model, trace, true_coeff,
              model_path, var_names=['age', 'gender', 'smoking', 'fever', 'vomiting', 'cutpoints'])
    make_plot(model, trace, prob,
              model_path + 'missing_data/', var_names=['w'])
