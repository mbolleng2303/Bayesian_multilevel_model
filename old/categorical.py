import pandas as pd
import theano
import theano.tensor as tt
import pymc3 as pm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


X, y = load_iris(True)
X_train, X_test, y_train, y_test = train_test_split(X, y)



Xt = theano.shared(X_train)
yt = theano.shared(y_train)

with pm.Model() as iris_model:

    # Coefficients for features
    β = pm.Normal("β", 0, sigma=1e2, shape=(4, 3))
    # Transoform to unit interval
    a = pm.Flat("a", shape=(3,))
    p = tt.nnet.softmax(Xt.dot(β) + a)

    observed = pm.Categorical("obs", p=p, observed=yt)