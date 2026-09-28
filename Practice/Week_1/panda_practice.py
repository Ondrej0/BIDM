import pandas as pd
import matplotlib.pyplot as plt


# Create a small fake dataset
data = {
    "employee": ["Tom", "Sarah", "James", "Lucy", "Ben", "Amy"],
    "department": ["IT", "Sales", "IT", "Sales", "IT", "Sales"],
    "salary": [30000, 35000, 32000, 40000, 38000, 37000]
}

df = pd.DataFrame(data)


# 1. Look at the DataFrame
print("1)")
print(df)


# 2. Select a column
print("2)")
print(df["salary"])


# 3. Filter rows
high_salary = df[df["salary"] > 35000]
print("3)")
print(high_salary)


# 4. Group data
average_salary = df.groupby("department")["salary"].mean()
print("4)")
print(average_salary)


# 5. Make a simple chart
average_salary.plot(kind="bar")

plt.title("Average Salary by Department")
plt.ylabel("Salary (£)")
plt.show()