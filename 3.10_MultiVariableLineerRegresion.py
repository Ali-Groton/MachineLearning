import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# y= a0+a1x +a2x2 ...... anxn -> mutli variable linear regression
# y= a0 + a1x + a2x2 

X = np.random.rand(100, 2)

coef = np.array([3,5])
# y = 0 + np.dot(X, coef)
y = 5 + 10*np.random.rand(100) + np.dot(X, coef)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection = "3d")
# ax.scatter(X[:, 0], X[:, 1], y)
# ax.set_xlabel("X1")
# ax.set_ylabel("X2")
# ax.set_zlabel("y")

lin_reg = LinearRegression()
lin_reg.fit(X,y)

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(X[:,0], X[:,1], y)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")

x1 ,x2 = np.meshgrid(np.linspace(0,1,10), np.linspace(0,1,10))
y_pred = lin_reg.predict(np.array([x1.flatten(),x2.flatten()]).T)
ax.plot_surface(x1, x2, y_pred.reshape(x1.shape),alpha = 0.3)
plt.title("multi variable lineer regression")

print("Katsaylılar : ", lin_reg.coef_)
print("Kesişim : ", lin_reg.intercept_)
