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
age = np.random.uniform(35, 65, N_sample).astype(int)
plt.hist(age, bins='auto')
plt.show()
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
outcome = generate_outcome (age, gender, smoking, fever, vomiting)
#plt.hist(outcome, bins='auto')
#plt.show()

data = list(zip(age, gender, smoking, fever, vomiting, outcome))

df = pd.DataFrame(data,
                  columns=['age', 'gender', 'smoking', 'fever', 'vomiting', 'outcome'])
df.to_csv('data.csv', index=False)


data = pd.read_csv('data.csv')

# correlation between  the features
seaborn.pairplot(data)
plt.show()

# Compute the correlation matrix
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = seaborn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
seaborn.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmax=0.3,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
    ax=ax,
)
plt.show()


