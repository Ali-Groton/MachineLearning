from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data
y = iris.target


pca = PCA(n_components = 2) #2 adet temel bileşen (PC)
X_pca = pca.fit_transform(X)

plt.figure()
for i in range(len(iris.target_names)):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label = iris.target_names[i])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of iris dataset")
plt.legend()

# %%
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target


pca = PCA(n_components = 3) #3 adet temel bileşen (PC)
X_pca = pca.fit_transform(X)  

fig = plt.figure(1, figsize = (8,6))
ax = fig.add_subplot(111, projection = "3d", elev = -150, azim = 110)

ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c = y, s = 40)

ax.set_title("First Three PCA  Dimensions of Irıs Datase")
ax.set_xlabel("1st Eigenvector")
ax.set_ylabel("2nd Eigenvector")
ax.set_zlabel("3rd Eigenvector")      


















