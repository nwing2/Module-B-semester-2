# %% [markdown]
# Nathalie Wing
# OMDS DX699 AI for Leaders

# %% [markdown]
# # Week 2 - Preprocessing, part 2
# 
# # 1. Lesson: None

# %% [markdown]
# # 2. Weekly graph question

# %% [markdown]
# The Storytelling With Data book mentions planning on a "Who, What, and How" for your data story.  Write down a possible Who, What, and How for your data, using the ideas in the book.

# %% [markdown]
# The Who, What, and How for the Valentines Spending Dataset (https://www.kaggle.com/datasets/aminasalamt/valentine-spending-dataset) is as follows:
# 
# WHO: the primary audience of the dataset is marketing and consumer insight teams at a retail company that wants actionable insights to inform targeted Valentine's Day promotions. Business stakeholder interested in seasonality of spending may also be interested in this dataset. This 'WHO' cares about spending trends, which demographic segments spend the most, and how spending categories differ by age and gender. The audience would want a clear visual with business recommendations based on this dataset, vs a more technical model. 
# 
# WHAT: They key story the dataset is trying to communicate is how Valentine's Day spending has changed over time, and how certain key demographics spend more on speciifc gift catgeories. The dataset contains data over multiple years and data across many product, age, and gender groups which can provide tailored insight into promotions and product offerings. 
# 
# HOW: This story can be communicated with visuals starting with a clear headline that reflects the key takeaway of the visuals produced to reflect the dataset. A possible visual includes a line graph that highlights spend trends over time, with Year on the X-axis and average spending per person on the Y-axis. This would show the overall increasing / decreasing trends and how overall spending hcanges across years. Another visual could look at age group spending preferences with a bar chart. This would compare spending categories by age groups, with age groups on the X-axis and % spending by category on the Y-axis, with the goal of highlighting which categories each age group prioritizes. A third visual could analyze gender-based differences. This could be visualized with a side-by-side bar chart of average % spending by category for men vs women. This can quickly highlight how spending preferences differ by gender. In the visualization, there should also be annotionas of key isnights directly on the visuals, and there should be a short Call to Action that drives an action as a result of findings in the visuals.

# %% [markdown]
# # 3. Homework - work with your own data

# %%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# %% [markdown]
# This week, you will do the same types of exercises as last week, but you should use your chosen datasets that someone in your class found last semester. (They likely will not be the particular datasets that you found yourself.)
# 
# ### Here are some types of analysis you can do  Use Google, documentation, and ChatGPT to help you:
# 
# - Summarize the datasets using info() and describe()
# 
# - Are there any duplicate rows?
# 
# - Are there any duplicate values in a given column (when this would be inappropriate?)
# 
# - What are the mean, median, and mode of each column?
# 
# - Are there any missing or null values?
# 
#     - Do you want to fill in the missing value with a mean value?  A value of your choice?  Remove that row?
# 
# - Identify any other inconsistent data (e.g. someone seems to be taking an action before they are born.)
# 
# - Encode any categorical variables (e.g. with one-hot encoding.)
# 
# ### Conclusions:
# 
# - Are the data usable?  If not, find some new data!
# 
# - Do you need to modify or correct the data in some way?
# 
# - Is there any class imbalance?  (Categories that have many more items than other categories).

# %%
# Import data from Kaggle

import os
os.system("kaggle datasets download -d aminasalamt/valentine-spending-dataset")
os.system("unzip -o valentine-spending-dataset.zip")
valentine = pd.read_csv("valentines_day_spending_dataset.csv")


# %%
# Get Datagframe Info & Describe
valentine.info()
valentine.describe(include='all')

# %%
# Check for Duplicates in each column

for col in valentine.columns:
    dup_count = valentine[col].duplicated().sum()
    print(f"Duplicate values in column '{col}': {dup_count}")

# %%
# Mean, Median, Mode

print("Mean:")
print(valentine.mean(numeric_only=True))

print("Median:")
print(valentine.median(numeric_only=True))

print("Mode:")
print(valentine.mode(numeric_only=True).iloc[0])

# %%
# Get Missing Values

valentine.isnull().sum()

# %%
# Fill missing values for gift_type column
valentine["gift_type"] = valentine["gift_type"].fillna("Unknown")

# Check nulls

valentine.isnull().sum()


# %%
# Check for inconsistent data 

numeric_cols = valentine.select_dtypes(include=np.number).columns

for col in numeric_cols:
    negative_count = (valentine[col] < 0).sum()
    print(f"Negative values in '{col}': {negative_count}")


# %%
# Encode Categorical Variables

categorical_cols = valentine.select_dtypes(include='object').columns
valentine_encoded = pd.get_dummies(valentine, columns=categorical_cols, drop_first=True)


# %%
# Class Imbalance Check

for col in categorical_cols:
    print(f"\nValue counts for {col}:")
    print(valentine[col].value_counts())

# %%
# Final cleaned & encoded datframe for preview:
print("\nFinal cleaned & encoded dataset preview:")
print(valentine_encoded.head())

# %% [markdown]
# # 4. Storytelling With Data graph

# %% [markdown]
# Just like last week: choose any graph in the Introduction of Storytelling With Data. Use matplotlib to reproduce it in a rough way. I don't expect you to spend an enormous amount of time on this; I understand that you likely will not have time to re-create every feature of the graph. However, if you're excited about learning to use matplotlib, this is a good way to do that. You don't have to duplicate the exact values on the graph; just the same rough shape will be enough.  If you don't feel comfortable using matplotlib yet, do the best you can and write down what you tried or what Google searches you did to find the answers.

# %% [markdown]
# The example used from Storytelling with Data is the before and after bar chart showing how decluttering and highlighting improves clarity. This example uses basic numbers to help illustrate visual shape and structure. The goal of this visualization is to reproduce a simple bar chart showing several categories, one category clearly highlighted, minimal gridlines, clear title, and direct labeling. This demonstrates the key idea from the books introduction that data visualization should emphasize the message, not deocration. 

# %%
import matplotlib.pyplot as plt

# Sample data 
categories = ['A', 'B', 'C', 'D', 'E']
values = [40, 65, 30, 85, 50]

# Create figure
plt.figure()

# Highlight one bar (like in the book)
colors = ['gray', 'gray', 'gray', 'steelblue', 'gray']

plt.bar(categories, values, color=colors)

# Add title (clear takeaway)
plt.title("Category D Significantly Outperforms Others")

# Remove unnecessary items on chart 
plt.xlabel("")
plt.ylabel("")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add annotation to highlight key insight
plt.text(3, 85, "Highest Value", ha='center')

plt.show()


