import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


learning_rate = [0.0001, 0.0005, 0.001, 0.005]
gamma = [0.96, 0.97, 0.98, 0.99]
epoch = [500, 1000, 1500, 2000, 2500, 3000]

data = pd.read_csv('scores.csv')

epochh = 3000

for_gamma = []
for lr in learning_rate:
    learning_gamma = []
    for gm in gamma:
        mask = (data['learning_rate'] == lr) & (data['gamma'] == gm) & (data['epoch'] == epochh)
        value = data.loc[mask, 'score'].values
        # print(value)
        learning_gamma.append(value)
    for_gamma.append(learning_gamma)

for value in for_gamma:
    for v in value:
        print(v)
# draw a legend plot
i = 0
for value in for_gamma:
    plt.plot(gamma, value, label='learning rate: {}'.format(learning_rate[i]))
    i += 1

plt.xlabel('gamma')
plt.ylabel('score')
plt.title('Score vs gamma varying learning rate')
plt.legend()
plt.show()

for_learning_rate = []
for gm in gamma:
    gamma_learning_rate = []
    for lr in learning_rate:
        mask = (data['learning_rate'] == lr) & (data['gamma'] == gm) & (data['epoch'] == epochh)
        value = data.loc[mask, 'score'].values
        # print(value)
        gamma_learning_rate.append(value)
    for_learning_rate.append(gamma_learning_rate)

for value in for_learning_rate:
    for v in value:
        print(v)

i = 0
for value in for_learning_rate:
    plt.plot(learning_rate, value, label='gamma: {}'.format(gamma[i]))
    i += 1

plt.xlabel('learning rate')
plt.ylabel('score')
plt.title('Score vs learning rate varying gamma')
plt.legend()
plt.show()

lr = 0.001
gm = 0.99
epoch = [500, 1000, 1500, 2000, 2500, 3000]
score = [18, 25, 205, 158, 16583, 65115]

plt.plot(epoch, score)
plt.xlabel('epoch')
plt.ylabel('score')
plt.title('Score vs epoch for lr={} and gm={}'.format(lr, gm))
plt.show()