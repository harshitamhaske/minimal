import numpy as np
import pandas as pd

# Define the file path
file_path = '' 

# Load the data 
data = np.load(file_path)

print("First 1000 rows of the data:")
print(data[:1000])

print("\nSummary Statistics:")
print("Mean of first column (Total Reward):", np.mean(data[:, 0]))
print("Mean of second column (Reward per Step):", np.mean(data[:, 1]))
print("Standard deviation of first column (Total Reward):", np.std(data[:, 0]))
print("Standard deviation of second column (Reward per Step):", np.std(data[:, 1]))

df = pd.DataFrame(data, columns=['Total Reward', 'Reward per Step'])

print("\nDataFrame Preview:")
print(df.head(1000))

csv_file_path = ''
df.to_csv(csv_file_path, index=False)
print(f"\nData saved to {csv_file_path}")

