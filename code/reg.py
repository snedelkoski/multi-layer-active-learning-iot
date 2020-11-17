from sklearn.datasets import make_friedman1
import matplotlib.pyplot as plt

data, y = make_friedman1(n_samples=10000, n_features=5, noise=0.0, random_state=None)

plt.plot(data[:,0].flatten())
plt.show()