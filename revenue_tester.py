from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_excel("retention_data.xlsx")

df = df.drop(['Unnamed: 0'], axis=1)

df['active_users'] = df['Install']

avg = df.mean()
avg = avg.drop(["Install", "Cost", "Revenue", "Install Day", "Day 14", "Day 21", "Day 30", "Day 45", "Day 60", "Day 90", "active_users"])

# calculate estimated active users
for i in range(len(df)-7):
    i = i + 7
    sum_temp = 0
    for j in range(len(avg)):
        sum_temp += avg.iloc[j]*df["Install"].iloc[i-j-1]/100
    df["active_users"].iloc[i] = sum_temp
    sum_temp = 0
    
    
df_1 = df.head(60)
df_2 = df.iloc[60:85]

y = df_1['Revenue']
x = df_1[['Cost','active_users']]
 
y_norm = (y-y.min())/(y.max()-y.min())
x_norm = (x-x.min())/(x.max()-x.min())

## Dummy df for estimated campaign 

# In here, we will take the values of b, p, and audience size (n) as input.
# Then, we will calculate estimated active users for each day of the campaign.
# Using active_users and cost as parameters, we will estimate revenue for each
# day using linear regression. We will cumsum daily revenues to find final 
# estimated total revenue.

model = LinearRegression().fit(x_norm,y_norm)

"""
#df2 = {'Install': p*n/7, 'Cost': b, 'active_users': 0}

for i in range(T):
    df = df.append(df2, ignore_index = True)
    


campaign_data = df[['Cost','active_users']].tail(T)
"""
# campaign_data_norm = (campaign_data-campaign_data.min())/(campaign_data.max()-campaign_data.min())

y_2 = df_2['Revenue'].reset_index()
x_2 = df_2[['Cost','active_users']]

x2_norm = (x_2-x_2.min())/(x_2.max()-x_2.min())

b = model.predict(x_2) # not using the normalized data, will be reviewed later
b_unnorm = b*(y.max()-y.min()) + y.min()

#total_revenue = b.sum()
#estimated_profit = total_revenue - B

# Create a new figure and axis
fig, ax = plt.subplots()

# Plot the lines
ax.plot(y_2.index, y_2['Revenue'], label='Observed Daily Revenue')
ax.plot(range(25), b, label='Estimated Daily Revenue')

# Set labels and title
ax.set_xlabel('Days')
ax.set_ylabel('Revenue')
ax.set_title('Observed vs. Estimated Daily Revenue')

# Display the legend
ax.legend()

# Show the plot
plt.show()



df_a = df[['Install', 'active_users']]

# Create a new figure and axis
fig, ax = plt.subplots()

# Plot the lines
ax.plot(df_a.index, df_a['Install'], label='Daily Number of Installs')
ax.plot(df_a.index, df_a['active_users'], label='Estimated Daily Users (from Retention)')

# Set labels and title
ax.set_xlabel('Days')
ax.set_ylabel('People')
ax.set_title('Installs vs. Estimated Daily Users')

# Display the legend
ax.legend()

# Show the plot
plt.show()