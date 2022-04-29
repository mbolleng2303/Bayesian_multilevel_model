'''import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt #for plotting

x, y, coef = datasets.make_regression(n_samples=100,#number of samples
                                      n_features=6,#number of features
                                      n_informative=6,#number of useful features
                                      noise=0,#bias and standard deviation of the guassian noise
                                      coef=True,#true coefficient used to generated the data
                                      random_state=0) #set for same data points for each run

# Scale feature x (years of experience) to range 0..20
x[:, 0] = np.interp(x, (x.min(), x.max()), (0, 20))

# Scale target y (salary) to range 20000..150000
y = np.interp(y, (y.min(), y.max()), (20000, 150000))

plt.ion() #interactive plot on
plt.plot(x,y,'.',label='training data')
plt.xlabel('Years of experience');plt.ylabel('Salary $')
plt.title('Experience Vs. Salary')'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
severity = np.zeros_like(vomiting)

plt.hist(severity, bins='auto')
# plt.show()

data = list(zip(age, gender, smoking, fever, vomiting, severity))

df = pd.DataFrame(data,
                  columns=['age', 'gender', 'smoking', 'fever', 'vomiting', 'severity'])
df.to_csv('data.csv', index=False)