import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("salary_data.csv")

# Features and target
X = data[["YearsExperience"]]
y = data["Salary"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict salary
experience = [[5]]
predicted_salary = model.predict(experience)

print("Predicted Salary for 5 years experience:", predicted_salary)

# Plot graph
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction using Linear Regression")
plt.show()
