from flask import Flask, flash, render_template, request, redirect, url_for, session, send_file
from utils import exponential_effectiveness, g_16
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

from constants import WORD_OF_MOUTH, INITIAL_EXPOSURE, ONLINE_LEARNING_N

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

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
            df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            df.to_pickle("./df.pickle")
            session['messages'] = filename
            return redirect('/rapor.html')
    return render_template("homepage.html")
    
@app.route('/rapor.html', methods =['GET', 'POST'])
def rapor():

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

    graphJSON = {}

    df = pd.read_pickle("./df.pickle")
    
    df = df.dropna()
    df = df[df['Campaign name'].str.contains('Android')]
    df = df.sort_values(by=['Day'])
    df['Kümülatif Harcanan Para'] = df['Amount spent (USD)'].cumsum()
    df['Kümülatif Ulaşılan Kişi Sayısı'] = df['Reach'].cumsum() / 500000
    cum_ad_spend_android = np.array(df['Kümülatif Harcanan Para'])
    cum_reach_android = np.array(df['Kümülatif Ulaşılan Kişi Sayısı'])
    df['Inferred'] = 'Ground Truth'
        
    popt, _ = curve_fit(exponential_effectiveness, cum_ad_spend_android, cum_reach_android)

    [m, u] = popt
    a = 2
    pareto_array = np.array([[p, t, *pareto_frontier_B_b(p, a, m, u, t)] for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] for t in np.linspace(1, 30, 30)])

    pareto_df = pd.DataFrame({'P': pareto_array[:, 0], 'T': pareto_array[:, 1], 'B': pareto_array[:, 2], 'Bid': pareto_array[:, 3]})
    pareto_df = pareto_df.dropna()
    
    for spend, reach in zip(cum_ad_spend_android, exponential_effectiveness(cum_ad_spend_android, *popt)):
        df = df.append({'Kümülatif Harcanan Para': spend, 'Kümülatif Ulaşılan Kişi Sayısı': reach, 'Inferred': 'Fit'}, ignore_index = True)
    
    graphJSON = plot_px(df)
    graphJSON2 = plot_pareto(pareto_df)    

    return render_template('rapor.html', graphJSON=graphJSON, graphJSON2=graphJSON2, m=m, u=u, kitle='Hedef Kitledeki Kişi Sayısı: 500000')

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
        df = df.dropna()
        df = df[df['Campaign name'].str.contains('Android')] # Android'e ozel olmamali
        df = df.sort_values(by=['Day'])
        df['Cumulative Spent'] = df['Amount spent (USD)'].cumsum()
        df['Cumulative Reach'] = df['Reach'].cumsum() / 500000 # Audience size belirlenmeli
        
        # Son n gunu kampanya_df ye ekle df['Amount spent (USD)'] df['Reach']            
        kampanya_df = pd.DataFrame(columns=['Day', 'B', 'T', 'Bid', 'P', 'M', 'U', 'Reach'])
        df2 = df.copy()
        df2.sort_values(by='Day', ascending=False)
        df2 = df2.tail(ONLINE_LEARNING_N)
        for i in range(ONLINE_LEARNING_N):
            new_entry = pd.DataFrame({'Day': 0, 'B': df2['Amount spent (USD)'].iloc[i], 'T': 0, 'Bid': 0, 'P': 0, 'M': 0, 'U': 0, 'Reach': df2['Reach'].iloc[i] / 500000}, index=[0])
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
            #parameters [m,u]
            #argss[g_star_arr, i, n]
            integral = scipy.integrate.quad(exponential_effectiveness, 0, i, args=(parameters[0], parameters[1]))
            summation = [(g_star_arr[j] - g_16(WORD_OF_MOUTH, INITIAL_EXPOSURE, integral)) for j in range(2)] #duzelt
            return (2 / ((ONLINE_LEARNING_N + 1) * ONLINE_LEARNING_N)) * np.sum(summation)       
    

        result = scipy.optimize.least_squares(step3, x0=np.array([kampanya_df.iloc[-1]['M'], kampanya_df.iloc[-1]['U']]), args=(g_star_arr, 5, ONLINE_LEARNING_N))
        yeni_bid = kampanya_df.iloc[-1]['Bid'] * 0.8
        kampanya_df = kampanya_df.append(pd.DataFrame({'Day': kampanya_df.iloc[-1]['Day'] + 1, 'B': yeni_b, 'T':  kampanya_df.iloc[-1]['T'], 'Bid': yeni_bid, 'P':  kampanya_df.iloc[-1]['P'], 'M': result.x[0], 'U': result.x[1], 'Reach':kampanya_df.iloc[-1]['Reach'] + float(request.form['Dün Alınan Sonuç'])/500000}), ignore_index=True)
        kampanya_df.to_pickle("./kampanya_df.pickle")
        flash('Yeni İhale Değeri: ' + str(yeni_bid))
    elif request.method == 'POST':
        flash('Lütfen formu doldurun')

    return render_template('kampanya.html', yeni_bid=yeni_bid)
    
if __name__ == "__main__":
    app.run(debug=True)
    
    
