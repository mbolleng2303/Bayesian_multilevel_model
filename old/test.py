import warnings
warnings.filterwarnings('ignore')
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


# Data
data_path = Path('data')
fig_path = Path('figures')
model_path = Path('models')
for p in [data_path, fig_path, model_path]:
    if not p.exists():
        p.mkdir()

data = pd.read_csv('data.csv')
data.info()
data.head()

# DataPreprocessing

''''categorical_features = data.select_dtypes(include=[np.object])

numeric_features = data.select_dtypes(include=[np.number])

cols = numeric_features.columns
numeric_features.loc[:, cols] = preprocessing.scale(numeric_features.loc[:, cols])

le = preprocessing.LabelEncoder()

categorical_features = data[categorical_features.columns].apply(lambda x: le.fit_transform(x))

data = pd.concat([categorical_features, numeric_features], axis=1)
data.info()'''

# see correlation
if False:
    plt.figure(figsize=(13, 13))
    corr = data.corr()
    mask = np.tri(*corr.shape).T
    sns.heatmap(corr.abs(), mask=mask, annot=True)
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.show()

    n_fts = len(data.columns)
    colors = cm.rainbow(np.linspace(0, 1, n_fts))

    data.drop('severity',axis=1).corrwith(data.severity).sort_values(ascending=True).plot(kind='barh',
                                                                                         color=colors, figsize=(12, 6))
    plt.title('Correlation to Target (severity)')
    plt.show()

    print('\n',data.drop('severity',axis=1).corrwith(data.severity).sort_values(ascending=False))

    data['smoking'] = -data['smoking']
    data['fever'] = -data['fever']
    data['age'] = -data['age']
    '''data['poutcome'] = -data['poutcome']
    data['loan'] = -data['loan']'''

    corr = data.corr()

    cor_target = corr["severity"]

    relevant_features = cor_target[cor_target > 0.08]
    relevant_features.sort_values(ascending=False)

    sns.pairplot(data[relevant_features.index])


# models
simple_model = "severity ~ age + gender + smoking"
full_model = "severity ~ age + gender + smoking + fever + vomiting"

''''with pm.Model() as manual_logistic_model:
    # random variables for coefficients with
    # uninformative priors for each parameter

    intercept = pm.Normal('intercept', 0, sd=100)
    beta_1 = pm.Normal('beta_1', 0, sd=100)
    beta_2 = pm.Normal('beta_2', 0, sd=100)
    beta_3 = pm.Normal('beta_3', 0, sd=100)

    # Transform random variables into vector of probabilities p(y_i=1)
    # according to logistic regression model specification.
    likelihood = pm.invlogit(intercept + beta_1 * data.age + beta_2 * data.gender + beta_3 * data.smoking)

    # Bernoulli random vector with probability of success
    # given by sigmoid function and actual data as observed
    pm.Bernoulli(name='logit', p=likelihood, observed=data.severity)

with manual_logistic_model:
    # compute maximum a-posteriori estimate
    # for logistic regression weights
    manual_map_estimate = pm.find_MAP()


def print_map(result):
    return pd.Series({k: np.asscalar(v) for k, v in result.items()})

print(print_map(manual_map_estimate))'''
''''
with pm.Model() as logistic_model:
    pm.glm.GLM.from_formula(simple_model,
                            data,
                            family=pm.glm.families.Binomial())
model = smf.logit(formula=simple_model, data=data[['severity', 'age', 'gender', 'smoking']])
result = model.fit()
print(result.summary())'''

with pm.Model() as logistic_model:
    pm.glm.GLM.from_formula(formula=full_model,
                            data=data,
                            family=pm.glm.families.Binomial())
print(logistic_model.basic_RVs)
with logistic_model:
    trace = pm.sample(tune=100,
                         draws=100,
                         chains=1,
                         init = 'adapt_diag',
                         cores=1)
pm.plot_trace(trace)
plt.show()

# save model
with open(model_path / 'logistic_model_nuts.pkl', 'wb') as buff:
    pickle.dump({'model': logistic_model, 'trace': trace}, buff)
print(pm.summary(trace))

with open(model_path / 'logistic_model_nuts.pkl', 'rb') as buff:
    data0 = pickle.load(buff)

logistic_model, trace_NUTS = data0['model'], data0['trace']

ppc = pm.sample_posterior_predictive(trace_NUTS, samples=500, model=logistic_model)
y_score = np.mean(ppc['y'], axis=0)
print(roc_auc_score(y_true=data.severity, y_score=y_score))




