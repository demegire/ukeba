from flask import Flask, flash, render_template, request, redirect, url_for, session
from utils import exponential_effectiveness, new_bid, pareto_frontier_B_b, maximize_p1p2_sum, maximize_n1n2_sum, maximize_ltv1ltv2_sum
from werkzeug.utils import secure_filename
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import json
import os

import plotly
import plotly.express as px

from constants import WORD_OF_MOUTH, INITIAL_EXPOSURE, ONLINE_LEARNING_N, AUDIENCE_SIZE, DECAY_RATE, P0

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
                df.to_pickle("./df.pickle")
                return redirect('/rapor.html')
            elif extension == 'xlsx':
                xl = pd.ExcelFile(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                if len(xl.sheet_names) == 1:

                    df = pd.read_excel(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    df = df.rename(columns={'Unnamed: 0': 'Date'})
                    df = df[df['Date'].notna()]
                    df['Cost'] = df['Cost'].str.replace('$', '').str.replace(',', '.') #1.000,23 seklinde . lar yok diye varsaydik
                    df['Cost'] = df['Cost'].astype(float)
                    df.to_pickle("./df.pickle")
                    return redirect('/rapor.html')
                else: # Install - Cost - Date columns are required in each sheet
                    for name in xl.sheet_names:
                        df = xl.parse(name)
                        if 'Unnamed: 0' in df.columns:
                            df = df.rename(columns={'Unnamed: 0': 'Date'})
                            df = df[df['Date'].notna()]
                            df['Cost'] = df['Cost'].str.replace('$', '').str.replace(',', '.') #1.000,23 seklinde . lar yok diye varsaydik
                        df['Cost'] = df['Cost'].astype(float)
                        df.to_pickle("./{}.pickle".format(name))
                    return redirect('/rapor_mp.html')
            
            session['messages'] = filename
            
    return render_template("homepage.html")
    
@app.route('/rapor.html', methods = ['GET', 'POST'])
def rapor():

    def plot_px(df):
        fig = px.line(df, x='Verilen İhale Değeri', y='Sonuç Yüzdesi', markers=True, color='Veri Kaynağı', title="Verilen İhale Değeri vs Sonuç Yüzdesi")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return graphJSON
    
    def plot_pareto(df):
        fig = px.line(df, x='B', y='T', markers=True, color='P', hover_data=['Bid'], title="B vs T vs P (Min B)")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return graphJSON

    graphJSON = {}

    df = pd.read_pickle("./df.pickle")

    df = df.sort_values(by=['Cost'])
    df['Verilen İhale Değeri'] = df['Cost'] 
    df['Sonuç Yüzdesi'] = df['Install'] / AUDIENCE_SIZE

    cum_ad_spend = np.array(df['Verilen İhale Değeri'], dtype='f')
    cum_result = np.array(df['Sonuç Yüzdesi'], dtype='f')
    df['Veri Kaynağı'] = 'Gerçek'
        
    popt, _ = curve_fit(exponential_effectiveness, cum_ad_spend, cum_result, p0=P0, maxfev=5000)

    [m, u] = popt
    pareto_array = np.array([[p, t, *pareto_frontier_B_b(p, WORD_OF_MOUTH, m, u, t, INITIAL_EXPOSURE)] for p in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.02, 0.03, 0.04, 0.05] for t in np.linspace(1, 30, 30)])

    pareto_df = pd.DataFrame({'P': pareto_array[:, 0], 'T': pareto_array[:, 1], 'B': pareto_array[:, 2], 'Bid': pareto_array[:, 3]})
    pareto_df = pareto_df.dropna()
    
    for spend, reach in zip(cum_ad_spend, exponential_effectiveness(cum_ad_spend, *popt)):
        df = df.append({'Verilen İhale Değeri': spend, 'Sonuç Yüzdesi': reach, 'Veri Kaynağı': 'Tahmin'}, ignore_index = True)
    
    graphJSON = plot_px(df)
    graphJSON2 = plot_pareto(pareto_df)    

    return render_template('rapor.html', graphJSON=graphJSON, graphJSON2=graphJSON2, m=m, u=u, kitle='Hedef Kitledeki Kişi Sayısı: ' + str(AUDIENCE_SIZE))


@app.route('/rapor_mp.html', methods =['GET', 'POST'])
def rapor_mp():
       
    # uploaded data 1
    
    #df1 = pd.read_pickle("./df_1.pickle")  
    df1 = pd.read_pickle("./df.pickle") 
    
    df1 = df1.sort_values(by=['Date'])
    df1['Verilen İhale Değeri'] = df1['Cost'].cumsum()
    df1['Sonuç Yüzdesi'] = df1['Install'].cumsum()  / AUDIENCE_SIZE

    cum_ad_spend_1 = np.array(df1['Verilen İhale Değeri'], dtype='f')
    cum_result_1 = np.array(df1['Sonuç Yüzdesi'], dtype='f')
    df1['Veri Kaynağı'] = 'Gerçek'
        
    popt_1, _ = curve_fit(exponential_effectiveness, cum_ad_spend_1, cum_result_1, p0=P0, maxfev=5000)


    # uploaded data 2
    
    #df2 = pd.read_pickle("./df_2.pickle")  
    df2 = pd.read_pickle("./df.pickle") 
    
    df2 = df2.sort_values(by=['Date'])
    df2['Verilen İhale Değeri'] = df2['Cost'].cumsum()
    df2['Sonuç Yüzdesi'] = df2['Install'].cumsum()  / AUDIENCE_SIZE

    cum_ad_spend_2 = np.array(df2['Verilen İhale Değeri'], dtype='f')
    cum_result_2 = np.array(df2['Sonuç Yüzdesi'], dtype='f')
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

        flash('Eski Bid: ' + str(bid))
         
        kampanya_df = pd.DataFrame(columns=['Day', 'B', 'T', 'Bid', 'P', 'M', 'U', 'Reach'])
        df2 = df.copy()
        for i in range(len(df)):
            new_entry = pd.DataFrame({'Date': df2['Date'].iloc[i], 'Day': 0, 'B': 0, 'T': 0, 'Bid': df2['Cost'].iloc[i], 'P': 0, 'M': 0, 'U': 0, 'Reach':  df2['Install'].iloc[i] / AUDIENCE_SIZE}, index=[0])            
            kampanya_df = kampanya_df.append(new_entry, ignore_index=True)
        kampanya_df = kampanya_df.append(pd.DataFrame({'Day': int(day), 'B': float(b), 'T': int(t), 'Bid': float(bid), 'P': float(p), 'M': float(m), 'U':float(u), 'Reach':0},  index=[0]), ignore_index=True)
        kampanya_df.to_pickle("./kampanya_df.pickle")
             
    kampanya_df = pd.read_pickle("./kampanya_df.pickle")
    
    print(kampanya_df)

    yeni_bid = 0

    flash('Kampanya Günü: ' + str(kampanya_df.iloc[-1]['T']))

    flash('Kalan Bütçe: ' + str(kampanya_df.iloc[-1]['B']))

    flash('Ulaşılan Kitle Oranı: ' + str(kampanya_df.iloc[-1]['Reach']))

    if request.method == 'POST' and  request.form['Dün Harcanılan Para'] and request.form['Dün Alınan Sonuç']:
        
        harcanilan_para = float(request.form['Dün Harcanılan Para'])
        sonuc = float(request.form['Dün Alınan Sonuç'])

        new_entry = pd.DataFrame({'Day': kampanya_df.iloc[-1]['Day'] + 1, 'B': 0, 'T': 0, 'Bid': harcanilan_para, 'P': 0, 'M': 0, 'U': 0, 'Reach':  sonuc / AUDIENCE_SIZE}, index=[0])
        
        kampanya_df = kampanya_df.sort_values(by=['Date'])
        gecici_df = kampanya_df.append(new_entry, ignore_index=True) # Gecici df m,u hesaplamak icin aciliyor
        gecici_df['Weights'] = np.array([np.power(DECAY_RATE, i) for i in range(len(gecici_df))]) # Exponential weighting, n>4000 de underflow sorunu olabilir

        gecici_df['Verilen İhale Değeri'] = gecici_df['Bid']
        gecici_df['Sonuç Yüzdesi'] = gecici_df['Reach']

        gecici_df = gecici_df.sort_values(by=['Bid'])
        cum_ad_spend = np.array(gecici_df['Verilen İhale Değeri'], dtype='f')
        cum_result = np.array(gecici_df['Sonuç Yüzdesi'], dtype='f')
            
        p0 = [kampanya_df.iloc[-1]['M'], kampanya_df.iloc[-1]['U']]

        popt, _ = curve_fit(exponential_effectiveness, cum_ad_spend, cum_result, p0=p0, maxfev=5000, sigma=gecici_df['Weights'].to_numpy())
        [m, u] = popt

        yeni_bid = new_bid(kampanya_df.iloc[-1]['P'], WORD_OF_MOUTH, m, u, kampanya_df.iloc[-1]['T'], kampanya_df.iloc[-1]['Reach'])
        flash('Yeni bid: ' + str(yeni_bid))
        if yeni_bid < 0:
             flash('Amaca ulaşıldı!')

        yeni_b = kampanya_df.iloc[-1]['B'] - harcanilan_para # Step 2
        if yeni_b < 0:
            flash('Bütçe aşıldı!')

        if yeni_bid > yeni_b:
            flash('Yeni ihale değeri, kalan bütçeyi aşıyor!')

        if kampanya_df.iloc[-1]['T'] == 1:
            flash('Kampanya süresi doldu!')

        kampanya_df = kampanya_df.append(pd.DataFrame({'Day': kampanya_df.iloc[-1]['Day'] + 1, 'B': yeni_b, 'T':  kampanya_df.iloc[-1]['T']-1, 'Bid': yeni_bid, 'P':  kampanya_df.iloc[-1]['P'], 'M': m, 'U': u, 'Reach': kampanya_df.iloc[-1]['Reach'] + (sonuc / AUDIENCE_SIZE)}, index=[0]), ignore_index=True)        
        kampanya_df.to_pickle("./kampanya_df.pickle")
        
    elif request.method == 'POST':
        flash('Lütfen formu doldurun')

    return render_template('kampanya.html', yeni_bid=yeni_bid)
    
if __name__ == "__main__":
    app.run(debug=True)
    
    
