from flask import Flask, flash, render_template, request, redirect, url_for, session, send_file
from utils import exponential_effectiveness, g_16, new_bid, pareto_frontier_B_b
from werkzeug.utils import secure_filename
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
import scipy
from io import BytesIO
#from snap7 import util
import pandas as pd
import numpy as np
import json
import os

import plotly
import plotly.express as px

from constants import WORD_OF_MOUTH, INITIAL_EXPOSURE, ONLINE_LEARNING_N, AUDIENCE_SIZE

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'pdf', 'xlsx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
@app.route('/')
def default():
    return redirect("homepage.html")
    
@app.route('/homepage.html', methods =['GET', 'POST'])
def index():
    if request.method == 'POST':
    # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            extension = filename.split('.')[-1]

            if extension == 'csv':
                df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            elif extension == 'xlsx':
                df = pd.read_excel(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                df = df.rename(columns={'Unnamed: 0': 'Date'})
                df = df[df['Date'].notna()]
                df['Cost'] = df['Cost'].str.replace('$', '').str.replace(',', '.') #1.000,23 seklinde . lar yok diye varsaydik
                df['Cost'] = df['Cost'].astype(float)
            df.to_pickle("./df.pickle")
            session['messages'] = filename
            return redirect('/rapor.html')
    return render_template("homepage.html")

@app.route('/rapor_kitle.html', methods = ['GET', 'POST'])
def rapor_kitle():
    
    graphJSON = {}
    df = pd.read_pickle("./df_kitle.pickle")
    #her kitle icin kumulatif ulasma ve harcama hesapla
    cumulatives = []
    popts = []

    for [spend, reach] in cumulatives:
        popts.append(curve_fit(exponential_effectiveness, spend, reach))

    return render_template('rapor_kitle.html')
    
@app.route('/rapor.html', methods = ['GET', 'POST'])
def rapor():

    def plot_px(df):
        fig = px.line(df, x='Kümülatif Harcanan Para', y='Kümülatif Sonuç Yüzdesi', markers=True, color='Veri Kaynağı', title="Harcanan Para vs Sonuç Yüzdesi")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return graphJSON
    
    def plot_pareto(df):
        fig = px.line(df, x='B', y='T', markers=True, color='P', hover_data=['Bid'], title="B vs T vs P (Min B)")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return graphJSON

    graphJSON = {}

    df = pd.read_pickle("./df.pickle")
    #df = df.head()

    #df = df.dropna()
    #df = df[df['Campaign name'].str.contains('Android')]
    #df = df.sort_values(by=['Day'])
    #df['Kümülatif Harcanan Para'] = df['Amount spent (USD)'].cumsum()
    #df['Kümülatif Sonuç Yüzdesi'] = df['Reach'].cumsum()

    df = df.sort_values(by=['Date'])
    df['Kümülatif Harcanan Para'] = df['Cost'].cumsum()
    df['Kümülatif Sonuç Yüzdesi'] = df['Install'].cumsum()  / AUDIENCE_SIZE

    cum_ad_spend = np.array(df['Kümülatif Harcanan Para'], dtype='f')
    cum_result = np.array(df['Kümülatif Sonuç Yüzdesi'], dtype='f')
    df['Veri Kaynağı'] = 'Gerçek'
        
    p0 = [9.42189734e+05, 2.19703087e-03]
    popt, _ = curve_fit(exponential_effectiveness, cum_ad_spend, cum_result, p0=p0, maxfev=5000)


    #popt = [0.014774, 0.913376, 0.632359, -0.014672]
    [m, u] = popt
    pareto_array = np.array([[p, t, *pareto_frontier_B_b(p, WORD_OF_MOUTH, m, u, t)] for p in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.02, 0.03, 0.04, 0.05] for t in np.linspace(1, 30, 30)])

    pareto_df = pd.DataFrame({'P': pareto_array[:, 0], 'T': pareto_array[:, 1], 'B': pareto_array[:, 2], 'Bid': pareto_array[:, 3]})
    pareto_df = pareto_df.dropna()
    
    for spend, reach in zip(cum_ad_spend, exponential_effectiveness(cum_ad_spend, *popt)):
        df = df.append({'Kümülatif Harcanan Para': spend, 'Kümülatif Sonuç Yüzdesi': reach, 'Veri Kaynağı': 'Tahmin'}, ignore_index = True)
    
    graphJSON = plot_px(df)
    graphJSON2 = plot_pareto(pareto_df)    

    return render_template('rapor.html', graphJSON=graphJSON, graphJSON2=graphJSON2, m=m, u=u, kitle='Hedef Kitledeki Kişi Sayısı: ' + str(AUDIENCE_SIZE))


@app.route('/rapor_multiplatform.html', methods =['GET', 'POST'])
def rapor_mp():

    def plot_px(df):
        fig = px.line(df, x='Kümülatif Harcanan Para', y='Kümülatif Ulaşılan Kişi Sayısı', markers=True, color='Inferred', title="Harcanan Para vs Ulaşılan Kişi Sayısı")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return graphJSON

    def plot_pareto(df):
        fig = px.line(df, x='B', y='T', markers=True, color='P', hover_data=['Bid'], title="B vs T vs P (Min B)")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return graphJSON

    def pareto_frontier_B_b(p, a, m, u, T):
        tmin = np.log((a * p + 1) / (1 - p)) / (m * (1 + a))
        B = ((-T/u) * np.log(1 - (tmin/T)))

        beta = u * (m - 1/((1 + a) * T) * np.log((a * p + 1)/((1 - p))))
        bid = 1 / u * np.log(m * u / beta)

        return B, bid
    
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
       


    
    # uploaded data 1
    
    #df1 = pd.read_pickle("./df_1.pickle")  
    df1 = pd.read_pickle("./df.pickle") 
    
    df1 = df1.sort_values(by=['Date'])
    df1['Kümülatif Harcanan Para'] = df1['Cost'].cumsum()
    df1['Kümülatif Sonuç Yüzdesi'] = df1['Install'].cumsum()  / AUDIENCE_SIZE

    cum_ad_spend_1 = np.array(df1['Kümülatif Harcanan Para'], dtype='f')
    cum_result_1 = np.array(df1['Kümülatif Sonuç Yüzdesi'], dtype='f')
    df1['Veri Kaynağı'] = 'Gerçek'
        
    p0 = [9.42189734e+05, 2.19703087e-03]
    popt_1, _ = curve_fit(exponential_effectiveness, cum_ad_spend_1, cum_result_1, p0=p0, maxfev=5000)


    # uploaded data 2
    
    #df2 = pd.read_pickle("./df_2.pickle")  
    df2 = pd.read_pickle("./df.pickle") 
    
    df2 = df2.sort_values(by=['Date'])
    df2['Kümülatif Harcanan Para'] = df2['Cost'].cumsum()
    df2['Kümülatif Sonuç Yüzdesi'] = df2['Install'].cumsum()  / AUDIENCE_SIZE

    cum_ad_spend_2 = np.array(df2['Kümülatif Harcanan Para'], dtype='f')
    cum_result_2 = np.array(df2['Kümülatif Sonuç Yüzdesi'], dtype='f')
    df2['Veri Kaynağı'] = 'Gerçek'
        
    p0 = [9.42189734e+05, 2.19703087e-03]
    popt_2, _ = curve_fit(exponential_effectiveness, cum_ad_spend_2, cum_result_2, p0=p0, maxfev=5000)


    # solve models for percentage, reach, ltv maximization


    b_limit = 5000 # take as input later
    platform_pars_1 = [0.1, 400000, 19.28] # a,n,ltv - take as input
    m_u_1 = popt_1
    platform_pars_2 = [2, 2500, 6.06]
    m_u_2 = popt_2
    
    p_results = maximize_p1p2_sum(platform_pars_1, m_u_1, platform_pars_2, m_u_2, b_limit=b_limit)
    reach_results = maximize_n1n2_sum(platform_pars_1, m_u_1, platform_pars_2, m_u_2, b_limit=b_limit)
    ltv_results = maximize_ltv1ltv2_sum(platform_pars_1, m_u_1, platform_pars_2, m_u_2, b_limit=b_limit)
    
    results = [p_results, reach_results, ltv_results]
    popts = [popt_1, popt_2]

    return render_template('rapor.html', kitle=results)



@app.route('/kampanya.html', methods =['GET', 'POST'])
def kampanya():
    day = request.args.get('day')
    b = request.args.get('b')
    t = request.args.get('t')
    bid = request.args.get('bid')
    p = request.args.get('p')
    m = request.args.get('m')
    u = request.args.get('u')

    if day == "1":
        df = pd.read_pickle('./df.pickle')

        #df = df.dropna()
        #df = df[df['Campaign name'].str.contains('Android')] # Android'e ozel olmamali
        #df = df.sort_values(by=['Day'])
        #df['Cumulative Spent'] = df['Amount spent (USD)'].cumsum()
        #df['Cumulative Reach'] = df['Reach'].cumsum() / 500000 # Audience size belirlenmeli
        
        # Son n gunu kampanya_df ye ekle df['Amount spent (USD)'] df['Reach']            
        kampanya_df = pd.DataFrame(columns=['Day', 'B', 'T', 'Bid', 'P', 'M', 'U', 'Reach'])
        df2 = df.copy()
        #df2.sort_values(by='Day', ascending=False)
        df2 = df2.tail(ONLINE_LEARNING_N)
        for i in range(ONLINE_LEARNING_N):
            new_entry = pd.DataFrame({'Day': 0, 'B': df2['Cost'].iloc[i], 'T': 0, 'Bid': 0, 'P': 0, 'M': 0, 'U': 0, 'Reach': df2['Install'].iloc[i] / AUDIENCE_SIZE}, index=[0])
            kampanya_df = kampanya_df.append(new_entry, ignore_index=True)
        kampanya_df = kampanya_df.append(pd.DataFrame({'Day': int(day), 'B': float(b), 'T': int(t), 'Bid': float(bid), 'P': float(p), 'M': float(m), 'U':float(u), 'Reach':0},  index=[0]), ignore_index=True)
        kampanya_df.to_pickle("./kampanya_df.pickle") # Yeni kampanya için overwrite
             
    kampanya_df = pd.read_pickle("./kampanya_df.pickle")

    yeni_bid = 0

    if request.method == 'POST' and  request.form['Dün Harcanılan Para'] and request.form['Dün Alınan Sonuç']:
        #kampanya_df ye ekle        
        g_star_arr = [] # Step 1
        for j in range(ONLINE_LEARNING_N): # Son n tane sample icin icin g_star hesapla
            total_effectiveness = 0
            for i in range(ONLINE_LEARNING_N):
                total_effectiveness += exponential_effectiveness(kampanya_df.iloc[-1 * (1 + j + i)]['Bid'], kampanya_df.iloc[-1]['M'], kampanya_df.iloc[-1]['U']) # Son n veriyi cek
            g_star_arr.append(g_16(WORD_OF_MOUTH, INITIAL_EXPOSURE, total_effectiveness))

        yeni_b = kampanya_df.iloc[-1] - float(request.form['Dün Harcanılan Para']) # Step 2
        
        def step3(parameters, g_star_arr, i, n):
            integral = scipy.integrate.quad(exponential_effectiveness, 0, i, args=(parameters[0], parameters[1]))
            summation = [(g_star_arr[j] - g_16(WORD_OF_MOUTH, INITIAL_EXPOSURE, integral)) for j in range(2)] #duzelt
            return (2 / ((ONLINE_LEARNING_N + 1) * ONLINE_LEARNING_N)) * np.sum(summation)       
    

        result = scipy.optimize.least_squares(step3, x0=np.array([kampanya_df.iloc[-1]['M'], kampanya_df.iloc[-1]['U']]), args=(g_star_arr, 5, ONLINE_LEARNING_N))
        yeni_bid = new_bid(kampanya_df.iloc[-1]['P'], WORD_OF_MOUTH, result.x[0], result.x[1], kampanya_df.iloc[-1]['T'])

        kampanya_df = kampanya_df.append(pd.DataFrame({'Day': kampanya_df.iloc[-1]['Day'] + 1, 'B': yeni_b, 'T':  kampanya_df.iloc[-1]['T'], 'Bid': yeni_bid, 'P':  kampanya_df.iloc[-1]['P'], 'M': result.x[0], 'U': result.x[1], 'Reach':kampanya_df.iloc[-1]['Reach'] + float(request.form['Dün Alınan Sonuç'])/500000}), ignore_index=True)
        kampanya_df.to_pickle("./kampanya_df.pickle")
        flash('Yeni İhale Değeri: ' + str(yeni_bid))
    elif request.method == 'POST':
        flash('Lütfen formu doldurun')

    return render_template('kampanya.html', yeni_bid=yeni_bid)
    
if __name__ == "__main__":
    app.run(debug=True)
    
    
