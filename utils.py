import numpy as np

def exponential_effectiveness(x, m, u):
    return m * (1 - np.exp(-1 * u * x))

def pareto_frontier_B_b(p, a, m, u, T, q):
    #tmin = np.log((a * p + 1) / (1 - p)) / (m * (1 + a))
    #B = ((-T/u) * np.log(1 - (tmin/T)))
    B = (T/u)*np.log( m*u / (u * (m - (1/((1+a)*T))*np.log((((a+1)*(1-q)/(1-p))-a*(1-q) ) / (1+a*q))) ) )
    bid = new_bid(p, a, m, u, T, q)

    return B, bid

def the_beta(p, a, m, u, T, q):
    return u * (m - 1/((1 + a) * T) * np.log((((a+1)*(1-q)/(1-p))-a*(1-q))/(1+a*q)) )

def new_bid(p, a, m, u, T, q):
    beta = the_beta(p, a, m, u, T, q)
    bid = 1 / u * np.log(m * u / beta)
    return bid

def pareto_plotter(b_par, T):
    plt.plot(np.arange(0, T, 0.1), b_par)
    plt.title("Pareto Frontier, B (y) vs T (x) ") 
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    return 0

def b_vs_p_plotter(b1_a, T):
    plt.plot(np.linspace(0,1,200), b1_a)
    plt.title("Pareto Frontier, B (y) vs p (x) for T={}".format(T))  
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    return 0

def maximize_p1p2_sum(platform_pars_1, m_u_1, platform_pars_2, m_u_2, b_limit, T=10, sens=10):
    max_sum = -1
    optimal_p1 = -1
    optimal_p2 = -1
    opt_b1 = -1
    opt_b2 = -1
    
    
    b1_a = [pareto_frontier_B_b(p,platform_pars_1[0],m_u_1[0],m_u_1[1],T)[0] for p in np.linspace(0,1,sens)]
    b2_a = [pareto_frontier_B_b(p,platform_pars_2[0],m_u_2[0],m_u_2[1],T)[0] for p in np.linspace(0,1,sens)]
    
    b1_array = np.column_stack((b1_a, np.linspace(0,1,sens)))
    b2_array = np.column_stack((b2_a, np.linspace(0,1,sens)))
    
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
    return optimal_p1, optimal_p2, opt_b1, opt_b2, max_sum

def maximize_n1n2_sum(platform_pars_1, m_u_1, platform_pars_2, m_u_2, b_limit, T=10, sens=10):
    max_sum = -1
    optimal_p1 = -1
    optimal_p2 = -1
    opt_b1 = -1
    opt_b2 = -1
    
    b1_a = [pareto_frontier_B_b(p,platform_pars_1[0],m_u_1[0],m_u_1[1],T)[0] for p in np.linspace(0,1,sens)]
    b2_a = [pareto_frontier_B_b(p,platform_pars_2[0],m_u_2[0],m_u_2[1],T)[0] for p in np.linspace(0,1,sens)]
    
    b1_array = np.column_stack((b1_a, platform_pars_1[1]*np.linspace(0,1,sens)))
    b2_array = np.column_stack((b2_a, platform_pars_2[1]*np.linspace(0,1,sens)))
    
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
    return optimal_p1, optimal_p2, opt_b1, opt_b2, max_sum

def maximize_ltv1ltv2_sum(platform_pars_1, m_u_1, platform_pars_2, m_u_2, b_limit, T=10, sens=10):
    max_sum = -1
    optimal_p1 = -1
    optimal_p2 = -1
    opt_b1 = -1
    opt_b2 = -1
    
    b1_a = [pareto_frontier_B_b(p,platform_pars_1[0],m_u_1[0],m_u_1[1],T)[0] for p in np.linspace(0,1,sens)]
    b2_a = [pareto_frontier_B_b(p,platform_pars_2[0],m_u_2[0],m_u_2[1],T)[0] for p in np.linspace(0,1,sens)]
    
    b1_array = np.column_stack((b1_a, platform_pars_2[2]*platform_pars_1[1]*np.linspace(0,1,sens)))
    b2_array = np.column_stack((b2_a, platform_pars_2[2]*platform_pars_2[1]*np.linspace(0,1,sens)))
    
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
    return optimal_p1, optimal_p2, opt_b1, opt_b2, max_sum

def b_calc(p, a, ms, T, ltv, n, disc):
    b1_a = [old_pareto_frontier(p,a,ms,T) for p in np.linspace(0,1,disc)]
    b1_array = np.column_stack((b1_a, ltv*n*np.linspace(0,1,disc)))  
    b_par = [old_pareto_frontier(p,a,ms,t) for t in np.arange(0, T, 0.1)]
    return b1_a, b1_array, b_par

def g_16(a, q, f_integral):
    return ((a + 1) * (1 - q)) / ((1 + a * q) * np.exp((1 + a) * (f_integral)) + a * (1 - q))
    