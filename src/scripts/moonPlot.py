import matplotlib.pyplot as plt
import numpy as np
import json

# Load the data
with open('plot_data.json') as f:
    plot_data = json.load(f)

xx = np.array(plot_data['xx'])
yy = np.array(plot_data['yy'])
Z = np.array(plot_data['Z'])
X = np.array(plot_data['X'])
y = np.array(plot_data['y'])


# h = 0.25
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))


print(Z.shape, xx.shape, yy.shape)
# Z = Z.reshape(xx.shape)

# Plotting
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.savefig('outputMoon.png')
