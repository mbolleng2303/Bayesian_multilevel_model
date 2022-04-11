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
from mlxtend.plotting import plot_confusion_matrix
import theano
import pymc3 as pm
from pymc3.variational.callbacks import CheckParametersConvergence
import statsmodels.formula.api as smf
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from IPython.display import HTML
warnings.filterwarnings('ignore')
np.random.seed(0)
data = pd.read_csv('data.csv')
data.info()
data.head()
X = data.drop('severity', axis=1)
y = data.severity
# models
simple_model = "severity ~ age + gender + smoking"
full_model = "severity ~ age + gender + smoking + fever + vomiting"
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
labels = X_train.columns
X_shared = theano.shared(X_train.values)
with pm.Model() as logistic_model_pred:
    pm.glm.GLM(x=X_shared, labels=labels,
               y=y_train, family=pm.glm.families.Binomial())
print(logistic_model_pred.basic_RVs)

with logistic_model_pred:
    pred_trace = pm.sample(draws=100,
                           tune=100,
                           chains=1,
                           cores=1,
                           init='adapt_diag')
X_shared.set_value(X_test)
ppc = pm.sample_posterior_predictive(pred_trace,
                    model=logistic_model_pred,
                    samples=100)

#pm.plot_trace(pred_trace)
#plt.show()
y_score = np.mean(ppc['y'], axis=0)
AUC = roc_auc_score(y_score=np.mean(ppc['y'], axis=0),
              y_true=y_test)

print("AUC for testset = ", AUC)
X_shared.set_value(X_train)
ppc_t = pm.sample_posterior_predictive(pred_trace,
                    model=logistic_model_pred,
                    samples=100)

#pm.plot_trace()
#plt.show()
y_score = np.mean(ppc_t['y'], axis=0)
AUC = roc_auc_score(y_score=np.mean(ppc_t['y'], axis=0),
              y_true=y_train)

print("AUC for trainset = ", AUC)
