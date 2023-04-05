def exponential_effectiveness(x, m, u):
    return m * (1 - np.exp(-1 * u * x))

def pareto_frontier_B_b(p, a, m, u, T):
    tmin = np.log((a * p + 1) / (1 - p)) / (m * (1 + a))
    B = ((-T/u) * np.log(1 - (tmin/T)))

    beta = u * (m - 1/((1 + a) * T) * np.log((a * p + 1)/((1 - p))))
    bid = 1 / u * np.log(m * u / beta)

    return B, bid

def old_pareto_frontier(p, a, ms, T):
    tmin = np.log((a * p + 1) / (1 - p)) / (ms * (1 + a))
    B = 100000000*((-T/popt[0]) * np.log(1 - (tmin/T)))
    return B

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

def maximize_p1p2_sum(b1_array, b2_array, a1, ms1, T1, a2, ms2, T2, b_limit):
    max_sum = -1
    optimal_p1 = -1
    optimal_p2 = -1
    opt_b1 = -1
    opt_b2 = -1
    for b1 in range(len(b1_array)):
        for b2 in range(len(b2_array)):
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
    