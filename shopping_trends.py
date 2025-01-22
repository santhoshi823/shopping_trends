from IPython import get_ipython
from IPython.display import display

from google.colab import files

uploaded = files.upload()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
filename = list(uploaded.keys())[0] 
df = pd.read_csv(io.BytesIO(uploaded[filename]))
# Distribution of customer ages
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=20, color='skyblue')
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
# Average purchase amount by category
avg_purchase_by_category = df.groupby('Category')['Purchase Amount (USD)'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
avg_purchase_by_category.plot(kind='bar', color='lightcoral')
plt.title('Average Purchase Amount by Category')
plt.xlabel('Product Category')
plt.ylabel('Average Purchase Amount (USD)')
plt.xticks(rotation=45)
plt.show()
# Count of purchases by gender
gender_purchase_count = df['Gender'].value_counts()
plt.figure(figsize=(6, 6))
gender_purchase_count.plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'pink'])
plt.title('Number of Purchases by Gender')
plt.ylabel('')
plt.show()
# Most commonly purchased items by category
most_purchased_items = df.groupby('Category')['Item Purchased'].value_counts().unstack().fillna(0)
print(most_purchased_items)
# Spending by season
season_spending = df.groupby('Season')['Purchase Amount (USD)'].sum()
plt.figure(figsize=(10, 6))
season_spending.plot(kind='bar', color='lightgreen')
plt.title('Total Spending by Season')
plt.xlabel('Season')
plt.ylabel('Total Spending (USD)')
plt.show()
# Average review rating by category
avg_rating_by_category = df.groupby('Category')['Review Rating'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
avg_rating_by_category.plot(kind='bar', color='lightblue')
plt.title('Average Review Rating by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Average Review Rating')
plt.xticks(rotation=45)
plt.show()
# Compare average purchase amounts between subscribed and non-subscribed customers
purchase_by_subscription = df.groupby('Subscription Status')['Purchase Amount (USD)'].mean()

plt.figure(figsize=(6, 6))
purchase_by_subscription.plot(kind='bar', color=['lightblue', 'lightcoral'])
plt.title('Average Purchase Amount by Subscription Status')
plt.xlabel('Subscription Status')
plt.ylabel('Average Purchase Amount (USD)')
plt.xticks(rotation=0)
plt.show()
# Count of purchases by payment method
payment_method_count = df['Payment Method'].value_counts()

plt.figure(figsize=(8, 6))
payment_method_count.plot(kind='bar', color='lightgreen')
plt.title('Payment Method Distribution')
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
# Compare average spending for customers who used promo codes vs those who didn't
spending_with_promo = df.groupby('Promo Code Used')['Purchase Amount (USD)'].mean()

plt.figure(figsize=(6, 6))
spending_with_promo.plot(kind='bar', color=['lightgreen', 'lightcoral'])
plt.title('Average Purchase Amount by Promo Code Usage')
plt.xlabel('Promo Code Used')
plt.ylabel('Average Purchase Amount (USD)')
plt.xticks(rotation=0)
plt.show()
# Group by age and calculate the frequency of purchases
age_groups = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65, 100], labels=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])

# Instead of calculating the mean, count the frequency of purchases for each age group
purchase_frequency_by_age = df.groupby(age_groups)['Frequency of Purchases'].count() 

# If you want the average number of purchases, you could first convert the frequency to numeric values.
# You would need a mapping like {'Fortnightly': 2, 'Weekly': 1, ...} and apply it to the column.

plt.figure(figsize=(10, 6))
purchase_frequency_by_age.plot(kind='bar', color='lightseagreen')
plt.title('Frequency of Purchases by Age Group')  # Changed title to reflect the count
plt.xlabel('Age Group')
plt.ylabel('Number of Purchases')  # Changed y-axis label to reflect the count
plt.show()
# Correlation between size and purchase amount (if 'Size' is numerical)
df['Size'] = pd.to_numeric(df['Size'], errors='coerce')  # Convert Size to numeric if it's not already
correlation = df[['Size', 'Purchase Amount (USD)']].corr()

sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation between Size and Purchase Amount')
plt.show()
# Group by category and shipping type, count the preferences
shipping_by_category = df.groupby(['Category', 'Shipping Type']).size().unstack().fillna(0)

shipping_by_category.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Shipping Type Preferences by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
# Compare average spending with and without discount
discount_effect = df.groupby('Discount Applied')['Purchase Amount (USD)'].mean()

plt.figure(figsize=(6, 6))
discount_effect.plot(kind='bar', color=['lightblue', 'lightcoral'])
plt.title('Average Purchase Amount with/without Discount')
plt.xlabel('Discount Applied')
plt.ylabel('Average Purchase Amount (USD)')
plt.xticks(rotation=0)
plt.show()
# Count of purchases by color
color_purchase_count = df['Color'].value_counts()

plt.figure(figsize=(8, 6))
color_purchase_count.plot(kind='bar', color='lightpink')
plt.title('Color Preferences of Customers')
plt.xlabel('Color')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
# Average number of previous purchases
avg_previous_purchases = df['Previous Purchases'].mean()

print(f"Average number of previous purchases: {avg_previous_purchases}")
# Average purchase amount by review rating
purchase_by_rating = df.groupby('Review Rating')['Purchase Amount (USD)'].mean()
plt.figure(figsize=(8, 6))
purchase_by_rating.plot(kind='bar', color='lightblue')
plt.title('Average Purchase Amount by Review Rating')
plt.xlabel('Review Rating')
plt.ylabel('Average Purchase Amount (USD)')
plt.xticks(rotation=0)
plt.show()
# Spending by location
location_spending = df.groupby('Location')['Purchase Amount (USD)'].sum()

plt.figure(figsize=(12, 8))
location_spending.plot(kind='bar', color='lightseagreen')
plt.title('Total Spending by Location')
plt.xlabel('Location')
plt.ylabel('Total Spending (USD)')
plt.xticks(rotation=45)
plt.show()
# Age group and product category relationship
age_groups = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65, 100], labels=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
age_category_counts = pd.crosstab(age_groups, df['Category'])

# Heatmap of age group vs category
sns.heatmap(age_category_counts, annot=True, cmap='Blues', fmt='d')
plt.title('Age Group vs Product Category')
plt.xlabel('Product Category')
plt.ylabel('Age Group')
plt.show()
# Compare average purchase amounts by gender
purchase_by_gender = df.groupby('Gender')['Purchase Amount (USD)'].mean()

plt.figure(figsize=(6, 6))
purchase_by_gender.plot(kind='bar', color=['lightblue', 'lightcoral'])
plt.title('Average Purchase Amount by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Purchase Amount (USD)')
plt.xticks(rotation=0)
plt.show()




