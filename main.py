import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("/Users/sujalkarki/Desktop/Stock_Market_Analysis/Stock.csv")
print(df.head(5))  #check first 5 row of the databases
print(df.isnull().sum()) # check missing value available or not 
print(df.shape) #check the shape of the dataset
print(df.info()) # check the information of the data set 

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# -----------------------------
#  Data Cleaning
# -----------------------------
# convert dataframe into Numpy array 
data = df[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy()
# fill missing values with 0
df.fillna(df.mean(), inplace=True)

# -------------------------------
# Detect outliers using Iqr method
# ------------------------------- 
def detect_outliers_iqr_np(data_np, feature_names=None):
 
    Q1 = np.percentile(data_np, 25, axis=0)
    Q3 = np.percentile(data_np, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((data_np < lower_bound) | (data_np > upper_bound))
    outlier_counts = np.sum(outliers, axis=0)

    if feature_names is None:
        feature_names = [f"Feature {i}" 
                         for i in range(data_np.shape[1])]

    print("🔍 Outlier Detection (Before Cleaning):")
    for i, count in enumerate(outlier_counts):
        print(f"- {feature_names[i]}: {count} outliers")

# Replace missing values (NaNs) with column means using NumPy
col_means = np.nanmean(data, axis=0)
inds = np.where(np.isnan(data))
data[inds] = np.take(col_means, inds[1])

# Run outlier detection
detect_outliers_iqr_np(data, feature_names=['Open', 'High', 'Low', 'Close', 'Volume'])

# ----------------------------------
#  Remove outliers using IQR method 
# ----------------------------------
def Iqr(data_np):
    Q1 = np.percentile(data_np, 25, axis=0)
    Q3 = np.percentile(data_np, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = np.all((data_np >= lower_bound) & (data_np <= upper_bound), axis=1)
    return data_np[mask]

clean_data = Iqr(data)

# Convert clean_data back to a DataFrame
clean_df = pd.DataFrame(clean_data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

# -----------------------------
#  Summary Statistics 
# -----------------------------
print(clean_df.describe())

# -----------------------------
#  Visualization
# -----------------------------

# Pie Chart: Outliers vs Non-Outliers
num_outliers = len(data) - len(clean_data)
labels = ['Clean Data', 'Outliers']
sizes = [len(clean_data), num_outliers]

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['Orange','Red'], startangle=90)
plt.title('Outliers vs Non-Outliers')
plt.axis('equal')
plt.show()

# ----------------------------------------
# Box Plot BEFORE Removing Outliers
# ----------------------------------------
plt.figure(figsize=(10, 5))
sns.boxplot(data=data, orient='h', palette='Set2')
plt.title("Box Plot - Before Removing Outliers")
plt.xlabel("Values")
plt.ylabel("Features")
plt.show()

# ----------------------------------------
# Box Plot AFTER Removing Outliers
# ----------------------------------------
plt.figure(figsize=(10, 5))
sns.boxplot(data=clean_data, orient='h', palette='Set3')
plt.title("Box Plot - After Removing Outliers")
plt.xlabel("Values")
plt.ylabel("Features")
plt.show()

# Convert cleaned NumPy data back to DataFrame for seaborn
clean_df = pd.DataFrame(clean_data, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

# Correlation Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(clean_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Line Plot – Close Price Over Time
plt.figure(figsize=(10, 5))
plt.plot(df['Date'][:len(clean_data)], clean_data[:, 3])  # Close prices
plt.title("Close Price Over Time (No Outliers)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()

# Distribution plot of Close Price
plt.figure(figsize=(8, 4))
sns.histplot(clean_df['Close'], kde=True, color='skyblue')
plt.title("Distribution of Close Prices")
plt.xlabel("Close Price")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Distribution plots
numeric_cols = clean_df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    plt.figure(figsize=(7, 4))
    sns.histplot(clean_df[col], kde=True, bins=30, color='steelblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


