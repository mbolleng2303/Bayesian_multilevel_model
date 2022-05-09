import warnings

from collections import OrderedDict
from multiprocessing import freeze_support
from time import time

import arviz as az
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn
import theano as thno
import theano.tensor as T

#from formulae import design_matrices
from scipy import integrate
from scipy.optimize import fmin_powell

print(f"Running on PyMC3 v{pm.__version__}")

warnings.filterwarnings("ignore")
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)





def plot_traces(traces, model, retain=0):
    """
    Convenience function:
    Plot traces with overlaid means and values
    """
    summary = az.summary(traces, stat_funcs={"mean": np.mean}, extend=False)
    ax = az.plot_trace(
        traces,
        lines=tuple([(k, {}, v["mean"]) for k, v in summary.iterrows()]),
    )

    for i, mn in enumerate(summary["mean"].values):
        ax[i, 0].annotate(
            f"{mn:.2f}",
            xy=(mn, 0),
            xycoords="data",
            xytext=(5, 10),
            textcoords="offset points",
            rotation=90,
            va="bottom",
            fontsize="large",
            color="C0",
        )



def create_poly_modelspec(k=1):
    """
    Convenience function:
    Create a polynomial modelspec string for patsy
    """
    return (
        "severity ~ age + gender + smoking + fever + vomiting" + " ".join([f"+ np.power(age,{j})" for j in range(2, k + 1)])
    ).strip()


raw_data = pd.read_csv(
    "data.csv")
# cleaning data
data = raw_data
with pm.Model() as logistic_model:
    pm.glm.GLM.from_formula(
        "severity ~ age + gender + smoking + fever + vomiting", data, family=pm.glm.families.Binomial()
    )

    trace = pm.sample(1000, tune=1000, init="adapt_diag")
plot_traces(trace, logistic_model)



