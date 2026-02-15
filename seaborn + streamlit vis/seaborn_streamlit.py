import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

st.title("Seaborn Visualization Dashboard on Tips Dataset By Amrutha Thalla")

# Set Seaborn style
sns.set_theme(style="whitegrid")

# Load the tips dataset
tips = sns.load_dataset('tips')

st.subheader("Dataset Shape")
st.write(tips.shape)
st.subheader("Dataset Preview")
st.write(tips)

# Scatter Plot
# Shows relationship between two numerical columns
fig1, ax1 = plt.subplots()
sns.scatterplot(data=tips, x='total_bill', y='tip',hue='sex', ax=ax1)
ax1.set_title("Scatter Plot: Total Bill vs Tip (Gender Split)")
st.pyplot(fig1)
plt.close(fig1)

# Line Plot
# Shows trend over time or order
fig2, ax2 = plt.subplots()
sns.lineplot(data=tips, x='size', y='total_bill',hue='sex', marker='o', ax=ax2)
ax2.set_title("Line Plot: Total Bill vs Tip (Colored by Gender)")
st.pyplot(fig2)
plt.close(fig2)

#Bar Plot
# Compares average values between categories
fig3, ax3 = plt.subplots()
sns.barplot(data= tips, x='day', y='total_bill',hue='sex', ax=ax3)
ax3.set_title("Bar Plot: Average Total Bill per Day (Male vs Female)")
st.pyplot(fig3)
plt.close(fig3)

# Box Plot
# Shows data distribution + outliers
fig4, ax4 = plt.subplots()
sns.boxplot(data=tips,x='day', y='tip',hue='smoker', ax=ax4)
ax4.set_title("Box Plot: Tip Distribution by Day (Smoker vs Non-Smoker)")
st.pyplot(fig4)
plt.close(fig4)

# Violin Plot
# Box plot + density shape combined
fig5, ax5 = plt.subplots()
sns.violinplot(data=tips, x='day', y='total_bill',hue='sex', ax=ax5)
ax5.set_title("Violin Plot: Total Bill Distribution by Day")
st.pyplot(fig5)
plt.close(fig5)

# Count Plot
# Shows frequency (how many)
fig6, ax6 = plt.subplots()
sns.countplot(data=tips, x='day',hue='smoker',ax=ax6)
ax6.set_title("Count Plot: Customers per Day")
st.pyplot(fig6)
plt.close(fig6)

#Regression Plot
# Scatter plot + best fit line
fig7, ax7 = plt.subplots()
sns.regplot(data=tips, x='total_bill', y='tip', ax=ax7)
ax7.set_title("Regression Plot: Total Bill vs Tip")
st.pyplot(fig7)
plt.close(fig7)

# Histogram
# Shows distribution of numerical data
fig8, ax8 = plt.subplots()
sns.histplot(data=tips, x='total_bill', kde=True, ax=ax8)
ax8.set_title("Histogram: Total Bill Distribution")
st.pyplot(fig8)
plt.close(fig8)

# Pair Plot
# Shows relationship between all numeric columns
st.subheader("Pair Plot")
pair_fig = sns.pairplot(tips, hue='sex')
st.pyplot(pair_fig)

# Cat Plot
# Advanced categorical plot (multiple graphs)
st.subheader("Cat Plot (Point Plot)")
cat_fig = sns.catplot(data=tips, x='day', y='tip', hue='sex', kind='point')
st.pyplot(cat_fig.fig)

