#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 05:39:59 2026

@author: nathaliewing
"""

# %% [markdown]
# # Week 9 - Multivariate Analysis, part 2

# %% [markdown]
# # 1. Lesson - no lesson this week

# %% [markdown]
# # 2. Weekly graph question

# %% [markdown]
# The three outliers in the upper-left corner of the PCA scatter plot have low first principal component (PC1) scores and high second principal component (PC2) scores. A low PC1 score indicates that these observations have relatively low overall values across all three series, since PC1 primarily captures the common variation shared by series_1, series_2, and series_3. In contrast, a high PC2 score reflects a strong deviation from that shared pattern. Because series_3 was constructed with substantially more noise than the other two series, PC2 is largely driven by how much series_3 differs from series_1 and series_2. Therefore, these outliers likely have relatively low values of series_1 and series_2, but comparatively higher values of series_3 than would be expected given the first two. It is much harder to say anything specific about the relationship between series_1 and series_2 for these points because the two variables are extremely similar by construction as series_2 is just series_1 with a small amount of noise, so they are highly correlated and contribute almost identically to PC1. As a result, PCA does not strongly distinguish between them, and any differences between the two are pushed into a minor component that is not visible in this two-dimensional plot. This makes it nearly impossible to infer their relative values from the graph. Overall, the PCA plot is useful for reducing dimensionality and highlighting general patterns, such as the dominant shared signal (PC1) and the additional variability introduced by series_3 (PC2), as well as making outliers easy to spot. However, this comes at the cost of interpretability, since the principal components are linear combinations of the original variables, and some detailed relationships—especially between highly correlated variables like series_1 and series_2—are obscured.

# %%
import numpy as np
import pandas as pd
from sklearn import decomposition
import matplotlib.pyplot as plt

np.random.seed(0)
num_points = 100
series_1 = np.random.normal(loc = 2, scale = 0.5, size = num_points)
series_2 = series_1 * (1 + np.random.normal(loc = 0, scale = 0.1, size = num_points))
series_3 = series_1 * (1 + np.random.normal(loc = 0, scale = 0.5, size = num_points))
df = pd.DataFrame({'ser1': series_1, 'ser2': series_2, 'ser3': series_3})
df = df - df.mean() # set mean to zero, so we don't have to subtract mean from the principal component scores

pca3 = decomposition.PCA(n_components = 3)
pca3.fit(df)
print(pca3.explained_variance_ratio_)
print(pca3.components_)

first_principal_component_score = df.dot(pca3.components_[0])
second_principal_component_score = df.dot(pca3.components_[1])
plt.scatter(first_principal_component_score, second_principal_component_score)
plt.xlabel("First Principal Component Score")
plt.ylabel("Second Principal Component Score")

# %% [markdown]
# The three outliers in the upper-left corner of the PCA scatter plot have low first principal component (PC1) scores and high second principal component (PC2) scores. A low PC1 score indicates that these observations have relatively low overall values across all three series, since PC1 primarily captures the common variation shared by series_1, series_2, and series_3. In contrast, a high PC2 score reflects a strong deviation from that shared pattern. Because series_3 was constructed with substantially more noise than the other two series, PC2 is largely driven by how much series_3 differs from series_1 and series_2. Therefore, these outliers likely have relatively low values of series_1 and series_2, but comparatively higher values of series_3 than would be expected given the first two. It is much harder to say anything specific about the relationship between series_1 and series_2 for these points because the two variables are extremely similar by construction as series_2 is just series_1 with a small amount of noise, so they are highly correlated and contribute almost identically to PC1. As a result, PCA does not strongly distinguish between them, and any differences between the two are pushed into a minor component that is not visible in this two-dimensional plot. This makes it nearly impossible to infer their relative values from the graph. Overall, the PCA plot is useful for reducing dimensionality and highlighting general patterns, such as the dominant shared signal (PC1) and the additional variability introduced by series_3 (PC2), as well as making outliers easy to spot. However, this comes at the cost of interpretability, since the principal components are linear combinations of the original variables, and some detailed relationships—especially between highly correlated variables like series_1 and series_2—are obscured.

# %% [markdown]
# # 3. Working on your datasets
# 
# This week, you will do the same types of exercises as last week, but you should use your chosen datasets that someone in your class found last semester. (They likely will not be the particular datasets that you found yourself.)
# 
# Here are some types of analysis you can do:
# Draw heatmaps.
# 
# Draw bubble plots.
# 
# Perform Principal Component Analysis to find out the directions in which the data varies.  Can you represent the data using only its projection onto its first principal component, using the methods described in Week 8?  How much of the variance would this capture?
# 
# Try performing linear regression analysis using different sets of features.  Which features seem most likely to be useful to predict other features?
# 
# Conclusions:
# Explain what conclusions you would draw from this analysis: are the data what you expect? Are the data likely to be usable? If the data are not useable, find some new data!
# 
# Do you see any outliers? (Data points that are far from the rest of the data).
# 
# Does the Principal Component Analysis suggest a way to represent the data using fewer dimensions than usual - using its first one or two principal component scores, perhaps?
# 
# Try using your correlation information from previous weeks to help choose features for linear regression.

# %% [markdown]
# ============================================
# VERSION 1: E-COMMERCE DATASET ANALYSIS
# ============================================

# %%
# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import kagglehub

# -----------------------------
# LOAD DATA (E-COMMERCE)
# -----------------------------
dataset_dir = kagglehub.dataset_download("ammaraahmad/us-ecommerce-record-2020")

print("Dataset directory:", dataset_dir)
print("Files found:", sorted(os.listdir(dataset_dir)))

csv_files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
csv_file = os.path.join(dataset_dir, csv_files[0])

df = pd.read_csv(csv_file, encoding="ISO-8859-1")

print("Shape:", df.shape)
print(df.head())
print(df.columns.tolist())

# %%
# -----------------------------
# CLEAN + SELECT NUMERIC DATA
# -----------------------------
numeric_cols = ['Sales', 'Profit', 'Quantity', 'Discount']
df_numeric = df[numeric_cols].dropna()

print(df_numeric.describe())

# %%
# -----------------------------
# 1. HEATMAP
# -----------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm")
plt.title("U.S. Ecom Dataset: Correlation Heatmap")
plt.show()

# %%
# -----------------------------
# 2. BUBBLE PLOT
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(
    df_numeric['Sales'],
    df_numeric['Profit'],
    s=df_numeric['Quantity'] * 10,
    alpha=0.5
)

plt.xlabel("Sales")
plt.ylabel("Profit")
plt.title("Bubble Plot (Size = Quantity)")
plt.show()

# %%
# -----------------------------
# 3. PCA
# -----------------------------
df_centered = df_numeric - df_numeric.mean()

pca = PCA()
pca.fit(df_centered)

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

pc_scores = pca.transform(df_centered)

plt.figure(figsize=(8,6))
plt.scatter(pc_scores[:,0], pc_scores[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Scatter Plot")
plt.show()

print("Variance captured by first PC:",
      pca.explained_variance_ratio_[0])

# %%
# -----------------------------
# 4. LINEAR REGRESSION
# -----------------------------
X = df_numeric[['Sales', 'Quantity', 'Discount']]
y = df_numeric['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R^2 score:", r2_score(y_test, y_pred))

coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print(coeff_df)

# %%
# -----------------------------
# 5. PAIRPLOT
# -----------------------------
sns.pairplot(df_numeric)
plt.show()

# %%
# -----------------------------
# 6. STORYTELLING WITH DATA
# -----------------------------
subcat_sales = df.groupby('Sub-Category')['Sales'].sum().sort_values()
plot_df = subcat_sales.reset_index()

highlight = plot_df.iloc[-1]['Sub-Category']

colors = [
    'lightgray' if cat != highlight else 'steelblue'
    for cat in plot_df['Sub-Category']
]

plt.figure(figsize=(10,6))
plt.barh(plot_df['Sub-Category'], plot_df['Sales'], color=colors)

plt.xlabel("Total Sales")
plt.title("Sales by Sub-Category (Highlighted Top Performer)")

sns.despine(left=True, bottom=True)

for i, (value, name) in enumerate(zip(plot_df['Sales'], plot_df['Sub-Category'])):
    if name == highlight:
        plt.text(value, i, f"  {name}", va='center')

plt.show()


# %% [markdown]
# ============================================
# VERSION 2: INSTACART DATASET ANALYSIS
# ============================================

# %%
# LOAD INSTACART DATA
dataset_dir = kagglehub.dataset_download(
    "psparks/instacart-market-basket-analysis",
    force_download=True
)

orders = pd.read_csv(
    os.path.join(dataset_dir, "orders.csv"),
    usecols=["order_id", "user_id", "order_number", "order_dow",
             "order_hour_of_day", "days_since_prior_order"],
    dtype={
        "order_id": "int32",
        "user_id": "int32",
        "order_number": "int16",
        "order_dow": "int8",
        "order_hour_of_day": "int8",
        "days_since_prior_order": "float32"
    }
)

prior = pd.read_csv(
    os.path.join(dataset_dir, "order_products__prior.csv"),
    usecols=["order_id", "product_id", "add_to_cart_order", "reordered"],
    dtype={
        "order_id": "int32",
        "product_id": "int32",
        "add_to_cart_order": "int16",
        "reordered": "int8"
    }
)

# SAMPLE USERS (avoid crash)
sample_users = orders["user_id"].drop_duplicates().sample(2500, random_state=42)

orders_sample = orders[orders["user_id"].isin(sample_users)]
prior_sample = prior[prior["order_id"].isin(orders_sample["order_id"])]

df = prior_sample.merge(orders_sample, on="order_id", how="left")

print("Instacart sample shape:", df.shape)

# %%
# -----------------------------
# CLEAN NUMERIC DATA
# -----------------------------
numeric_cols = [
    "add_to_cart_order",
    "reordered",
    "order_number",
    "order_dow",
    "order_hour_of_day",
    "days_since_prior_order"
]

df_numeric = df[numeric_cols].dropna()

print(df_numeric.describe())

# %%
# HEATMAP
corr = df_numeric.corr()

# Mask upper triangle (cleaner look)
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",                # cleaner numbers
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 10}
)

plt.title(
    "Instacart Feature Correlation Heatmap\n"
    "(Weak Pairwise Relationships Across Behavioral Variables)",
    fontsize=13
)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

# %%
# BUBBLE PLOT

plt.figure(figsize=(10,7))

scatter = plt.scatter(
    df["order_number"],
    df["days_since_prior_order"],
    s=df["add_to_cart_order"] * 5,   # slightly larger bubbles
    c=df["reordered"],
    cmap="coolwarm",                 # better contrast for binary
    alpha=0.3,                       # visible but not overcrowded
    edgecolors="none"
)

plt.xlabel("Order Number (Customer Purchase Sequence)", fontsize=11)
plt.ylabel("Days Since Prior Order", fontsize=11)

plt.title(
    "Customer Ordering Behavior and Reorder Patterns\n"
    "(Color = Reordered, Size = Cart Position)",
    fontsize=13
)

# Add colorbar legend
cbar = plt.colorbar(scatter)
cbar.set_label("Reordered (0 = No, 1 = Yes)")

# Optional: add grid for readability
plt.grid(alpha=0.2)

plt.tight_layout()
plt.show()

# %%
# PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Keep reordered column BEFORE scaling
df_pca = df_pca = df_numeric.copy()
df_pca["reordered"] = df["reordered"]

# Drop rows with missing values consistently
df_pca = df_pca.dropna()

# Split features + color
X = df_pca.drop(columns=["reordered"])
y = df_pca["reordered"]

# Scale + PCA
scaler = StandardScaler()
X_scaled = X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
pc_scores = pca.fit_transform(X_scaled)

# Extract variance explained
variance_explained = pca.explained_variance_ratio_

# Plot
plt.figure(figsize=(10,7))

scatter = plt.scatter(
    pc_scores[:,0],
    pc_scores[:,1],
    c=y,
    cmap="coolwarm",
    alpha=0.1,
    edgecolors="none"
)

plt.colorbar(scatter, label="Reordered (0 = No, 1 = Yes)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    f"PCA with Reorder Behavior\n"
    f"PC1 explains {variance_explained[0]:.2%}, "
    f"PC2 explains {variance_explained[1]:.2%}"
)

plt.show()

# %%
# LINEAR REGRESSION
X = df_numeric[["add_to_cart_order", "reordered", "order_number",
                "order_dow", "order_hour_of_day"]]
y = df_numeric["days_since_prior_order"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R^2:", r2_score(y_test, y_pred))

# %%
# PAIRPLOT (sampled for performance)
sns.pairplot(df_numeric.sample(min(1000, len(df_numeric))), corner=True)
plt.show()

# %%
# STORYTELLING PLOT
plot_df = (
    df.groupby("order_dow")["reordered"]
      .sum()
      .sort_values()
      .reset_index()
)

highlight = plot_df.iloc[-1]["order_dow"]

colors = [
    "lightgray" if day != highlight else "steelblue"
    for day in plot_df["order_dow"]
]

plt.figure(figsize=(10,6))
plt.barh(plot_df["order_dow"], plot_df["reordered"], color=colors)

plt.title("Reorders by Day of Week (Highlight Top)")
sns.despine()

plt.show()