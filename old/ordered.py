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
model_path = "../ordered/"
N_sample = 100
# patient id
patient_id = np.linspace(0, N_sample, N_sample).astype(int)
# age
mu = 50
sigma = 5
age = np.random.uniform(35, 65, N_sample).astype(int)
#plt.hist(age, bins='auto')
# plt.show()
# gender
gender = np.random.choice([0, 1], N_sample, p=[0.5, 0.5])
#plt.hist(gender, bins='auto')
# plt.show()

# Smoking
smoking = np.random.choice([0, 1, 2, 3], N_sample, p=[0.3, 0.3, 0.3, 0.1])
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
    #noise = np.random
    for i in range(age.shape[0]):
        score[i] += (10/8)*age[i]
        if gender[i] == 1 :#male
            score[i] += 22
        if smoking[i] != 0 :# dont konow
            score[i]+= 37 + smoking[i]*8
        if fever[i]==1 :
            score[i]+= 34
        if vomitting[i] == 0 :
            score[i]+= 45
    A = np.percentile(score, 33)
    B = np.percentile(score, 66)
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
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.01,
                                                    random_state=42)
#labels = X_train.columns

X_train = X
y_train = y
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
    #a = pm.Normal("intercept", mu=0, sigma=1.0)  # intercepts
    b = pm.Normal("age", mu=0, sigma=1.5)
    c = pm.Normal("gender", mu=0, sigma=1.5)
    d = pm.Normal("smoking", mu=0, sigma=1.5)
    e = pm.Normal("fever", mu=0, sigma=1.5)
    f = pm.Normal("vomiting", mu=0, sigma=1.5)
    # association of income with choice
    phi = b * X_shared[:, 0] + c * X_shared[:, 1] + d * X_shared[:, 2] + e * X_shared[:, 3] + f * X_shared[:, 4]

    outcome = pm.OrderedLogistic("outcome", eta=phi, cutpoints=cutpoints, observed=y_train)

    trace_ordered_multinomial = pm.sample(draws=1000,
                           tune=1000,
                           chains=4,
                           cores=1,
                           init='adapt_diag')

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