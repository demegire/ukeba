import plotly.express as px
import pandas as pd

# Define the populations
h1 = 10000
h2 = 5000

# Define the solutions for budget
B1 = 150
B2 = 200

# Define the solutions for exposure percentage
p1 = 0.05
p2 = 0.10

# Define platform names
names_pl = ['Meta', 'Ironsource']

# Define the data
data = {
    'budget_allocated': [B1, B2],
    'exposure_percentage': [p1, p2],
    'exposed_population': [h1*p1, h2*p2],
    'names_platform': names_pl,
    'unexposed_population': [h1*(1-p1), h2*(1-p2)],
    'platform_1_exposed_unexposed':[h1*p1, h1*(1-p1)],
    'platform_2_exposed_unexposed':[h2*p2, h2*(1-p2)],
    'names_exposure': ['Exposed Population', 'Unexposed Population']
       
}

# Create the DataFrame
df = pd.DataFrame(data)

# Pie for budget allocation
fig = px.pie(df, values='budget_allocated', names='names_platform', title='Budgets to be allocated')
fig.show()

# Pie for exposure in platform 1
fig = px.pie(df, values='platform_1_exposed_unexposed', names='names_exposure', title=names_pl[0])
fig.show()

# Pie for exposure in platform 2
fig = px.pie(df, values='platform_2_exposed_unexposed', names='names_exposure', title=names_pl[1])
fig.show()