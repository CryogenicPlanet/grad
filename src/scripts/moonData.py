import json
from sklearn.datasets import make_moons, make_blobs
import random
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1337)
random.seed(1337)


X, y = make_moons(n_samples=100, noise=0.1)

y = y*2 - 1  # make y be -1 or 1
# visualize in 2D
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='jet')


plt.savefig('inputMoon.png')


# Convert dataset to list for JSON serialization
data = {'X': X.tolist(), 'y': y.tolist()}

# Write dataset to a JSON file
with open('moonData.json', 'w') as json_file:
    json.dump(data, json_file)
