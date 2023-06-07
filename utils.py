import numpy as np
from sklearn.linear_model import LinearRegression

def exponential_effectiveness(x, m, u):
    return m * (1 - np.exp(-1 * u * x))

def pareto_frontier_B_b(p, a, m, u, T, q):    
    B = (T/u)*np.log( m*u / (u * (m - (1/((1+a)*T))*np.log((((a+1)*(1-q)/(1-p+1e-10))-a*(1-q) ) / (1+a*q)))))
    bid = new_bid(p, a, m, u, T, q)
    return B, bid

def beta_func(p, a, m, u, T, q):
    return u * (m - 1/((1 + a) * T) * np.log((((a+1)*(1-q)/(1-p))-a*(1-q))/(1+a*q)) )

def new_bid(p, a, m, u, T, q):
    beta = beta_func(p, a, m, u, T, q)
    bid = 1 / u * np.log(m * u / beta)
    return bid

def maximize_ltv1ltv2_sum(platform_pars_1, m_u_1, platform_pars_2, m_u_2, b_limit, T=10, sens=1000):
    max_sum = -1
    optimal_p1 = -1
    optimal_p2 = -1
    opt_b1 = -1
    opt_b2 = -1
    current_sum = -1
    p_vals = np.linspace(0.0000001, 1, num=1000)
    
    p_vals_to_ltv1 = np.linspace(0.0000001, 1, num=1000)#[0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]
    p_vals_to_ltv2 = np.linspace(0.0000001, 1, num=1000)#[0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]
    
    for pp in range(len(p_vals)):
        p_vals_to_ltv1[pp] = platform_pars_1[2]*platform_pars_1[1]*p_vals[pp]
    
    for pp in range(len(p_vals)):
        p_vals_to_ltv2[pp] = platform_pars_2[2]*platform_pars_2[1]*p_vals[pp]
    
    
    b1_a = [pareto_frontier_B_b(p,platform_pars_1[0],m_u_1[0],m_u_1[1],T,q=0)[0] for p in p_vals]
    b2_a = [pareto_frontier_B_b(p,platform_pars_2[0],m_u_2[0],m_u_2[1],T,q=0)[0] for p in p_vals]
    
    b1_array = np.column_stack((b1_a, p_vals_to_ltv1))
    b2_array = np.column_stack((b2_a, p_vals_to_ltv2))
    
    for b1 in range(len(b1_a)):
        for b2 in range(len(b2_a)):
            if (b1_array[b1][0]+b2_array[b2][0])<b_limit:
                p1 = b1_array[b1][1]
                p2 = b2_array[b2][1]
                current_sum = p1 + p2
            if current_sum > max_sum:
                max_sum = current_sum
                optimal_p1 = p1
                optimal_p2 = p2
                opt_b1 = b1_array[b1][0]
                opt_b2 = b2_array[b2][0]
    
    actual_p1 = optimal_p1/(platform_pars_1[2]*platform_pars_1[1])
    actual_p2 = optimal_p2/(platform_pars_2[2]*platform_pars_2[1])
    
    actual_reach1 = optimal_p1/platform_pars_1[2]
    actual_reach2 = optimal_p2/platform_pars_2[2]
    
    actual_ltv1 = optimal_p1
    actual_ltv2 = optimal_p2
    
    return actual_p1, actual_p2, actual_reach1, actual_reach2, actual_ltv1, actual_ltv2, opt_b1, opt_b2, max_sum, 

def data_organizer(df): #toyshop data
    df = df.drop(['eCPI', 'Imp.(UA)', 'eCPM(UA)', 'Profit', 'ROI', 'Paying Users', 'IAP Count', 'Last Day'], axis=1)
    df = df.fillna(method='ffill')
    df = df.replace('[\$]', '', regex=True)
    df = df.replace('[\%]', '', regex=True)
    df = df.replace('[\.]', '', regex=True)
    df = df.replace(',', '.', regex=True)
    
    arpu_index = []
    n = 0
    for i in range(int(len(df)/3)):
        arpu_index.append(3*i)
        n = n+1
    
    retention_index = arpu_index
    retention_index = [index + 1 for index in arpu_index]
    users_index = [index + 2 for index in arpu_index]
    
    df_arpu = df.iloc[arpu_index]
    df_arpu['Install Day'] = df_arpu['Install Day'].str.replace('ARPU: ', '')
    df_arpu.iloc[:, 1:] = df_arpu.iloc[:, 1:].astype(float)
    df_arpu = df_arpu.tail(-1).reset_index(drop=True)
    
    df_retention = df.iloc[retention_index]
    df_retention['Install Day'] = df_retention['Install Day'].str.replace('Retention: ', '')
    df_retention.iloc[:, 1:] = df_retention.iloc[:, 1:].astype(float)
    df_retention = df_retention.tail(-1).reset_index(drop=True)
    
    df_users = df.iloc[users_index]
    df_users['Install Day'] = df_users['Install Day'].str.replace('Users: ', '')
    df_users.iloc[:, 1:] = df_users.iloc[:, 1:].astype(float)
    df_users = df_users.tail(-1).reset_index(drop=True)
    
    return df_retention
    
def revenue_estimator(df, p, n, T, B, b):
    
    df = data_organizer(df)
    df = df.drop(['Unnamed: 0'], axis=1)

    df['active_users'] = df['Install']
    
    avg = df.mean()
    avg = avg.drop(["Install", "Cost", "Revenue", "Install Day", "Day 14", "Day 21", "Day 30", "Day 45", "Day 60", "Day 90", "active_users"])
    
    y = df['Revenue']
    x = df[['Cost','active_users']]
     
    y_norm = (y-y.min())/(y.max()-y.min())
    x_norm = (x-x.min())/(x.max()-x.min())
    
    ## Dummy df for estimated campaign 
    
    # In here, we will take the values of b, p, and audience size (n) as input.
    # Then, we will calculate estimated active users for each day of the campaign.
    # Using active_users and cost as parameters, we will estimate revenue for each
    # day using linear regression. We will cumsum daily revenues to find final 
    # estimated total revenue.

    model = LinearRegression().fit(x_norm,y_norm)
    
    df2 = {'Install': p*n/7, 'Cost': b, 'active_users': 0}
    
    for i in range(T):
        df = df.append(df2, ignore_index = True)
        
    # calculate estimated active users
    for i in range(len(df)-7):
        i = i + 7
        sum_temp = 0
        for j in range(len(avg)):
            sum_temp += avg.iloc[j]*df["Install"].iloc[i-j-1]/100
        df["active_users"].iloc[i] = sum_temp
        sum_temp = 0
    
    campaign_data = df[['Cost','active_users']].tail(T)
    campaign_data_norm = (campaign_data-campaign_data.min())/(campaign_data.max()-campaign_data.min())
    b = model.predict(campaign_data) # not using the normalized data, will be reviewed later
    b_unnorm = b*(y.max()-y.min()) + y.min()
    
    total_revenue = b.sum()
    estimated_profit = total_revenue - B
    
    return estimated_profit