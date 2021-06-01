import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import seaborn as sns

from Algorithm import compute_cost
from Algorithm import gradient_decent
from Algorithm import predict

X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1, random_state=14)

y = y[:,np.newaxis]

sns.set_style('white')
sns.scatterplot(X[:,0],X[:,1],hue=y.reshape(-1));

m=len(y)

X = np.hstack((np.ones((m,1)),X))

n=np.size(X,1)
params=np.zeros((n,1))

iterations=1500
learning_rate=0.03

initial_cost = compute_cost(X, y, params)
(cost_history, params_optimal) = gradient_decent(X, y, params, learning_rate, iterations)


plt.figure()
sns.set_style('white')
plt.plot(range(len(cost_history)), cost_history, 'r')
plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()


y_pred=predict(X,params_optimal)
score=float(sum(y_pred==y))/float(len(y))


slope = -(params_optimal[1] / params_optimal[2])
intercept = -(params_optimal[0] / params_optimal[2])

sns.set_style('white')
sns.scatterplot(X[:,1],X[:,2],hue=y.reshape(-1));

ax = plt.gca()
ax.autoscale(False)
x_vals = np.array(ax.get_xlim())
y_vals = intercept + (slope * x_vals)
plt.plot(x_vals, y_vals, c="k")
plt.show()