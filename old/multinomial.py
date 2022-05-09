import numpy as np
import pandas as pd
import seaborn as seaborn
from matplotlib import pyplot as plt

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
N_sample = 100
# patient id
patient_id = np.linspace(0, N_sample, N_sample).astype(int)
# age
mu = 50
sigma = 5
age = np.random.normal(mu, sigma, N_sample).astype(int)
#plt.hist(age, bins='auto')
# plt.show()
# gender
gender = np.random.choice([0, 1, 3, 4], N_sample, p=[0.46, 0.46, 0.04, 0.04])
#plt.hist(gender, bins='auto')
# plt.show()

# Smoking
smoking = np.random.choice([0, 1, 2, 8], N_sample, p=[0.3, 0.3, 0.3, 0.1])
#plt.hist(smoking, bins='auto')
# plt.show()

# Fever
fever = np.random.choice([0, 1], N_sample, p=[0.5, 0.5])
#plt.hist(fever, bins='auto')
# plt.show()

# Vomiting
vomiting = np.random.choice([0, 1], N_sample, p=[0.5, 0.5])
#plt.hist(vomiting, bins='auto')
# plt.show()

def generate_outcome (age, gender, smoking, fever, vomitting):
    score = np.zeros_like(age)
    outcome = np.zeros_like(age)
    for i in range(age.shape[0]):
        score[i] += (10/8)*age[i]
        if gender[i] == 1 :#male
            score[i] += 22
        if smoking[i]== 8 :# dont konow
            score[i]+= 62
        if smoking[i] == 1:# former
            score[i] += 45
        if smoking[i] == 0:# never
            score[i] += 52
        if fever[i]==1 :
            score[i]+= 34
        if vomitting[i] == 0 :
            score[i]+= 45
    A = np.percentile(score, 60)
    B = np.percentile(score, 85)
    C = np.percentile(score, 100)
    for i in range(score.shape[0]):
        if score[i] <= A:
            outcome[i] = 0
        elif score[i] >= B:
            outcome[i] = 2
        else :
            outcome[i] = 1
    return outcome
# Severity
outcome = generate_outcome (age, gender, smoking, fever, vomiting)#np.random.choice([0, 1, 2], N_sample, p=[0.6, 0.25, 0.15])  # np.random.choice([0, 1], N_sample, p=[0.75, 0.25])

#plt.hist(outcome, bins='auto')
#plt.show()

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
with pm.Model() as multinomial:
    a = pm.Normal("a", mu=0, sigma=1.0, shape=3)  # intercepts
    b = pm.Normal("b", mu=0, sigma=1.5, shape=3)
    c = pm.Normal("c", mu=0, sigma=1.5, shape=3)
    d = pm.Normal("d", mu=0, sigma=1.5, shape=3)
    e = pm.Normal("e", mu=0, sigma=1.5, shape=3)
    f = pm.Normal("f", mu=0, sigma=1.5, shape=3)
    # association of income with choice
    s0 = a[0] + b[0] * X_shared[:, 0] + c[0] * X_shared[:, 1] + d[0] * X_shared[:, 2] + e[0] * X_shared[:, 3] + f[0] * X_shared[:, 4]
    s1 = a[1] + b[1] * X_shared[:, 0] + c[1] * X_shared[:, 1] + d[1] * X_shared[:, 2] + e[1] * X_shared[:, 3] + f[1] * X_shared[:, 4]
    s2 = a[2] + b[2] * X_shared[:, 0] + c[2] * X_shared[:, 1] + d[2] * X_shared[:, 2] + e[2] * X_shared[:, 3] + f[2] * X_shared[:, 4]
    #s2 = a[2] + np.zeros(X_shared[:, 0].shape[0]) #pivoting the intercept for the third category
    s = pm.math.stack([s0, s1, s2]).T
    p_ = tt.nnet.softmax(s)
    outcome_obs = pm.Categorical("outcome", p=p_, observed=y_train)

    trace_multinomial = pm.sample(draws=4,
                           tune=2,
                           chains=1,
                           cores=1,
                           init='adapt_diag')

X_shared.set_value(X_train)
ppc_t = pm.sample_posterior_predictive(trace_multinomial,
                    model=multinomial,
                    samples=10)

y_score = np.round(np.mean(ppc_t['outcome'], axis=0))
acc = accuracy_score(y_train, y_score)
print("acc for trainset = ", acc)

X_shared.set_value(X_test)
ppc = pm.sample_posterior_predictive(trace_multinomial,
                    model=multinomial,
                    samples=10)

# pm.plot_trace(trace_multinomial)
# plt.show()
y_score = np.round(np.mean(ppc_t['outcome'], axis=0))
acc = accuracy_score(y_train, y_score)
print("acc for testset = ", acc)
#az.summary(trace_multinomial)

