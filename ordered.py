import warnings
from pathlib import Path
import pickle
from collections import OrderedDict
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
warnings.filterwarnings('ignore')
np.random.seed(0)
N_sample = 10
# patient id
patient_id = np.linspace(0, N_sample, N_sample).astype(int)
# age
mu = 50
sigma = 5
age = np.random.normal(mu, sigma, N_sample).astype(int)
plt.hist(age, bins='auto')
# plt.show()
# gender
gender = np.random.choice([0, 1, 3, 4], N_sample, p=[0.46, 0.46, 0.04, 0.04])
plt.hist(gender, bins='auto')
# plt.show()

# Smoking
smoking = np.random.choice([0, 1, 2, 8], N_sample, p=[0.3, 0.3, 0.3, 0.1])
plt.hist(smoking, bins='auto')
# plt.show()

# Fever
fever = np.random.choice([0, 1], N_sample, p=[0.5, 0.5])
plt.hist(fever, bins='auto')
# plt.show()

# Vomiting
vomiting = np.random.choice([0, 1], N_sample, p=[0.5, 0.5])
plt.hist(vomiting, bins='auto')
# plt.show()

# Severity
outcome = np.random.choice([0, 1, 2], N_sample, p=[0.6, 0.25, 0.15])  #np.random.choice([0, 1], N_sample, p=[0.75, 0.25])#
plt.hist(outcome, bins='auto')
# plt.show()

data = list(zip(age, gender, smoking, fever, vomiting, outcome))

df = pd.DataFrame(data,
                  columns=['age', 'gender', 'smoking', 'fever', 'vomiting', 'outcome'])
df.to_csv('data.csv', index=False)


data = pd.read_csv('data.csv')
data.info()
data.head()
X = data.drop('outcome', axis=1)
y = data.outcome
# models
simple_model = "outcome ~ age + gender + smoking"
full_model = "outcome ~ age + gender + smoking + fever + vomiting"
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
labels = X_train.columns
X_shared = theano.shared(X_train.values)

X_train = np.array(X_train)
nbr_variable = X_train.shape[1]
nbr_classes = 3
with pm.Model() as ordered_multinomial:

    cutpoints = pm.Normal(
        "cutpoints",
        0.0,
        1.5,
        transform=pm.distributions.transforms.ordered,
        shape=nbr_classes,
        testval=np.arange(nbr_classes) - (nbr_classes-1)/2,
    )
    a = pm.Normal("a", mu=0, sigma=10, shape=1)  # intercepts
    b = pm.Normal("b", mu=0, sigma=15, shape=1)
    c = pm.Normal("c", mu=0, sigma=15, shape=1)
    d = pm.Normal("d", mu=0, sigma=15, shape=1)
    e = pm.Normal("e", mu=0, sigma=15, shape=1)
    # association of income with choice
    phi = pm.Deterministic("phi", a[0] + b[0] * X_shared[:, 0] + c[0] * X_shared[:, 1] + d[0] * X_shared[:, 2] + e[0] * X_shared[:, 3])

    outcome = pm.OrderedLogistic("outcome", phi, cutpoints, observed = y_train)

    trace_ordered_multinomial = pm.sample(draws=4,
                           tune=10,
                           chains=2,
                           cores=1,
                           init='adapt_diag')

X_shared.set_value(X_train)
ppc_t = pm.sample_posterior_predictive(trace_ordered_multinomial,
                    model=ordered_multinomial,
                    samples=10)

y_score = np.round(np.mean(ppc_t['outcome'], axis=0))
acc = accuracy_score(y_train, y_score)
print("AUC for trainset = ", acc)

X_shared.set_value(X_test)
ppc = pm.sample_posterior_predictive(trace_ordered_multinomial,
                    model=ordered_multinomial,
                    samples=10)

# pm.plot_trace(trace_multinomial)
# plt.show()
y_score = np.round(np.mean(ppc_t['outcome'], axis=0))
acc = accuracy_score(y_train, y_score)
print("acc for testset = ", acc)
#az.summary(trace_multinomial)