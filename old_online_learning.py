def online():
            last_index = len(kampanya_df) #i
        yeni_q = int(request.form['Dün Alınan Sonuç']) / 2500000 + kampanya_df.iloc[-1]['Reach']
        #kampanya_df ye ekle        
        g_models = [] # Step 1,
        for j in range(last_index - ONLINE_LEARNING_N + 1, last_index): # Son n tane sample icin icin g_star hesapla
            total_effectiveness = 0
            for k in range(ONLINE_LEARNING_N): 
                total_effectiveness += exponential_effectiveness(kampanya_df.iloc[j-k]['Bid'], kampanya_df.iloc[-1]['M'], kampanya_df.iloc[-1]['U']) # Son n veriyi cek
            g_models.append(g_16(WORD_OF_MOUTH, yeni_q, total_effectiveness)) # Q = o gun alinan sonuc / yeni kitle
        print('G Models')
        print(g_models)
        yeni_b = kampanya_df.iloc[-1]['B'] - float(request.form['Dün Harcanılan Para']) # Step 2
        if yeni_b < 0:
            flash('Out of budget')
        def step3(parameters):
            integral = lambda t: ((parameters[0] * parameters[1] - the_beta(kampanya_df.iloc[-1]['P'], WORD_OF_MOUTH, parameters[0], parameters[1], kampanya_df.iloc[-1]['T'])) * t / parameters[1]) #scipy.integrate.quad(exponential_effectiveness, 0, i, args=(parameters[0], parameters[1]))
            summation = [((j - last_index + ONLINE_LEARNING_N) * (g_models[j - last_index + ONLINE_LEARNING_N - 1] - g_16(WORD_OF_MOUTH, yeni_q, integral(j)))) for j in range(last_index - ONLINE_LEARNING_N + 1, last_index)] #bircok seyi duzelt
            return (2 / ((ONLINE_LEARNING_N + 1) * ONLINE_LEARNING_N)) * np.sum(summation)   
        
        """
        mm_range = np.linspace(0, 5, 100)
        uu_range = np.linspace(0, 5, 100)
        MM, UU = np.meshgrid(mm_range, uu_range)
        
        # evaluate the function for each combination of parameters
        z = np.zeros_like(MM)
        for i in range(len(mm_range)):
            for j in range(len(uu_range)):
                z[i,j] = step3([MM[i,j], UU[i,j]])

        # create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(MM, UU, z, cmap='coolwarm')
        ax.set_xlabel('mm')
        ax.set_ylabel('uu')
        ax.set_zlabel('function output')
        plt.savefig('function_surface.png')
        """
            
    
        result = scipy.optimize.least_squares(step3, x0=np.array([kampanya_df.iloc[-1]['M'], kampanya_df.iloc[-1]['U']]), gtol=1e-10, xtol=1e-31, ftol=1e-31)
        print('Opt Results')
        print(result)
        yeni_bid = new_bid(kampanya_df.iloc[-1]['P'], WORD_OF_MOUTH, result.x[0], result.x[1], kampanya_df.iloc[-1]['T'])
        kampanya_df = kampanya_df.append(pd.DataFrame({'Day': kampanya_df.iloc[-1]['Day'] + 1, 'B': yeni_b, 'T':  kampanya_df.iloc[-1]['T'], 'Bid': yeni_bid, 'P':  kampanya_df.iloc[-1]['P'], 'M': result.x[0], 'U': result.x[1], 'Reach':yeni_q}, index=[0]), ignore_index=True)        
        print('Yeni Kampanya Df')
        print(kampanya_df)
        kampanya_df.to_pickle("./kampanya_df.pickle")
        flash('Yeni İhale Değeri: ' + str(yeni_bid))