import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
data_size = 1000

data = pd.DataFrame({
    'size': np.random.randint(500, 5000, size=data_size),  
    'bedrooms': np.random.randint(1, 6, size=data_size), 
    'bathrooms': np.random.randint(1, 4, size=data_size),  
    'location_index': np.random.randint(1, 6, size=data_size),
    'amenities_score': np.random.uniform(1, 10, size=data_size),  
    'price': np.random.randint(50000, 500000, size=data_size)  
})

X = data[['size', 'bedrooms', 'bathrooms', 'location_index', 'amenities_score']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30, color='blue')
plt.xlabel("Residuals")
plt.title("Residuals Distribution")
plt.show()
