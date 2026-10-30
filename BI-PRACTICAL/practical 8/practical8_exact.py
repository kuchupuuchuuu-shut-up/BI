
# ---------------- STEP 1: Import Libraries ----------------
import pandas as pd
import matplotlib.pyplot as plt


# ---------------- STEP 2: Load the CSV File ----------------
df = pd.read_csv("practical8.csv")
print(df.head())


# ---------------- STEP 3: Check Dataset Shape ----------------
print(df.shape)


# ---------------- STEP 4: Display Column Names ----------------
print(df.columns)


# ---------------- STEP 5: Check Missing Values ----------------
print(df.isnull().sum())


# ---------------- STEP 6: Count Records by Species ----------------
print(df["species"].value_counts())


# ---------------- STEP 7: Visualize Species Distribution ----------------
df["species"].value_counts().plot(kind="bar")
plt.title("Number of Flowers in Each Species")
plt.xlabel("Species")
plt.ylabel("Number of Flowers")
plt.show()
