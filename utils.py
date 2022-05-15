import pickle
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
import time

def make_plot(model, trace, true_coeff, model_path, var_names = None):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print(model.basic_RVs)

    az.plot_trace(trace, var_names=var_names, combined=False)
    plt.savefig(model_path + 'trace')

    az.summary(trace).to_csv(model_path + 'trace.csv')
    print(az.summary(trace))

    az.plot_pair(trace, kind=["scatter", "kde"], point_estimate="median", var_names= var_names)
    plt.savefig(model_path + 'plot_pair')

    az.plot_energy(trace)
    plt.savefig(model_path + 'energy')

    az.plot_posterior(trace, var_names=var_names, ref_val=np.array(true_coeff).tolist())
    plt.savefig(model_path + 'posterior')

    az.plot_forest(trace, var_names=var_names)
    for j, (y_tick, frac_j) in enumerate(zip(plt.gca().get_yticks(), reversed(true_coeff))):
        plt.vlines(frac_j, ymin=y_tick - 0.45, ymax=y_tick + 0.45, color="black", linestyle="--")
    plt.savefig(model_path + 'forest', dpi=100)

