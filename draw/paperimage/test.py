import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data from log file
log_file_path = './evaluation/32.log'
with open(log_file_path, 'r') as file:
    lines = file.readlines()

# Remove header and parse data
data_lines = lines[1:]
data = [line.strip().split(',') for line in data_lines]

# Convert data to DataFrame
df = pd.DataFrame(data, columns=['name', 'x', 'y', 'labelx', 'labely', 'error'])

# Convert relevant columns to numeric
df['x'] = pd.to_numeric(df['x'])
df['y'] = pd.to_numeric(df['y'])
df['labelx'] = pd.to_numeric(df['labelx'])
df['labely'] = pd.to_numeric(df['labely'])

# Scatter plot 1: labelx vs x
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(df['labelx'], df['x'], color='blue', label='Data points')
plt.plot(df['labelx'], df['labelx'], color='red', linestyle='--', label='y = x')

# Linear regression fit
reg1 = LinearRegression().fit(df['labelx'].values.reshape(-1, 1), df['x'].values)
x_fit = np.linspace(df['labelx'].min(), df['labelx'].max(), 100)
y_fit = reg1.predict(x_fit.reshape(-1, 1))
plt.plot(x_fit, y_fit, color='green', label='Fitted line')

plt.xlabel('Ground Truth x')
plt.ylabel('Estimation x')
plt.title('x-axis Scatter Plot')
plt.legend()
plt.grid(True)

# Scatter plot 2: labely vs y
plt.subplot(1, 2, 2)
plt.scatter(df['labely'], df['y'], color='blue', label='Data points')
plt.plot(df['labely'], df['labely'], color='red', linestyle='--', label='y = x')

# Linear regression fit
reg2 = LinearRegression().fit(df['labely'].values.reshape(-1, 1), df['y'].values)
y_fit2 = reg2.predict(np.linspace(df['labely'].min(), df['labely'].max(), 100).reshape(-1, 1))
plt.plot(np.linspace(df['labely'].min(), df['labely'].max(), 100), y_fit2, color='green', label='Fitted line')


plt.xlabel('Ground Truth y')
plt.ylabel('Estimation y')
plt.title('y-axis Scatter Plot')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.savefig('scatter_plot.png')
plt.show()
