from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 ,random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

y_pred = lin_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred) 
rmse = np.sqrt(mse) # Manuel olarak karekök al
# rmse = mean_squared_error(y_test, y_pred, squared = False)
print("rmse: ", rmse)


