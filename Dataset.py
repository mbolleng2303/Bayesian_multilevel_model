import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
import tensorflow_probability as tfp


class SimulatedData:

    def __init__(self, n_sample, outcome='severity level'):
        self.type = outcome
        self.N = n_sample
        self.age = np.random.uniform(0, 8, n_sample).astype(int)
        self.gender = np.random.choice([0, 1], n_sample, p=[0.5, 0.5])
        self.smoking = np.random.choice([0, 1, 2, 3], n_sample, p=[0.3, 0.3, 0.3, 0.1])
        self.fever = np.random.choice([0, 1], n_sample, p=[0.5, 0.5])
        self.vomiting = np.random.choice([0, 1], n_sample, p=[0.5, 0.5])
        # possible other variable
        #  self.race = np.random.choice([0, 1], N_sample, p=[0.5, 0.5])
        #  self.bmi = np.random.uniform(5, 50, N_sample).astype(int)
        if self.type == 'severity level':
            self.outcome = self.generate_ordered_outcome()
        elif self.type == 'ICU':
            self.outcome = self.generate_binary_outcome()
        self.nbr_classes = np.unique(self.outcome).shape[0]
        self.data = list(zip(self.age, self.gender, self.smoking, self.fever, self.vomiting, self.outcome))

        df = pd.DataFrame(self.data,
                          columns=['age', 'gender', 'smoking', 'fever', 'vomiting', 'outcome'])
        df.to_csv('data.csv', index=False)

        self.data = pd.read_csv('data.csv')
        self.data.info()
        self.data.head()

    def generate_binary_outcome(self):

        score = np.zeros_like(self.age)
        outcome = np.zeros_like(self.age)
        # noise = np.random
        for i in range(self.age.shape[0]):
            score[i] += (10 / 8) * (30 + self.age[i] * 5)
            if self.gender[i] == 1:
                score[i] += 22
            if self.smoking[i] != 0:
                score[i] += 37 + self.smoking[i] * 8
            if self.fever[i] == 1:
                score[i] += 34
            if self.vomiting[i] == 0:
                score[i] += 45

        tresh = np.percentile(score, 66)
        for i in range(score.shape[0]):
            prob = np.array(tfp.distributions.OrderedLogistic(cutpoints=[float(tresh)],
                                                              loc=float(score[i])).categorical_probs())
            outcome[i] = np.random.choice([0, 1], 1, p=prob)
        return outcome

    def generate_ordered_outcome(self):

        score = np.zeros_like(self.age)
        outcome = np.zeros_like(self.age)
        # noise = np.random
        for i in range(self.age.shape[0]):
            score[i] += (10 / 8) * (30+self.age[i]*5)
            if self.gender[i] == 1:
                score[i] += 22
            if self.smoking[i] != 0:
                score[i] += 37 + self.smoking[i] * 8
            if self.fever[i] == 1:
                score[i] += 34
            if self.vomiting[i] == 0:
                score[i] += 45
        A = np.percentile(score, 33)
        B = np.percentile(score, 66)

        for i in range(score.shape[0]):
            prob = np.array(tfp.distributions.OrderedLogistic(cutpoints=[float(A), float(B)],
                                                              loc=float(score[i])).categorical_probs())
            outcome[i] = np.random.choice([0, 1, 2], 1, p=prob)

            '''if score[i] <= A:
                outcome[i] = 0
            elif score[i] >= B:
                outcome[i] = 2
            else:
                outcome[i] = 1'''
        return outcome

    def explore_dataset(self):

        data = self.data
        plt.savefig('none.png')
        seaborn.pairplot(data)
        plt.savefig('data_pairplot.png')

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
        plt.savefig('data_correlation.png')

    def __getitem__(self, item):
        if item == 'x':
            return self.data.drop('outcome', axis=1)
        elif item == 'y':
            return self.data.outcome
        else:
            print('error')
