from flask import Flask, flash, render_template, request, redirect, url_for, session, send_file
from utils import exponential_effectiveness, g_16
from werkzeug.utils import secure_filename
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
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
    
@app.route('/homepage.html')
def index():
    return render_template("homepage.html")
    
    
@app.route('/yukle.html', methods =['GET', 'POST'])
def yukle():
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
            
    return render_template('yukle.html')
    
@app.route('/rapor.html', methods =['GET', 'POST'])
def rapor():

    def plot_px(df):
        fig = px.line(df, x='Cumulative Spent', y='Cumulative Reach', markers=True, color='Inferred', title="Reach vs Spent")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return graphJSON

    def plot_pareto(df):
        fig = px.line(df, x='B', y='T', markers=True, color='P', hover_data=['Bid'], title="B vs T Pareto")
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
    df['Cumulative Spent'] = df['Amount spent (USD)'].cumsum()
    df['Cumulative Reach'] = df['Reach'].cumsum() / 500000
    cum_ad_spend_android = np.array(df['Cumulative Spent'])
    cum_reach_android = np.array(df['Cumulative Reach'])
    df['Inferred'] = 'Ground Truth'
        
    popt, _ = curve_fit(exponential_effectiveness, cum_ad_spend_android, cum_reach_android)

    [m, u] = popt
    a = 2
    pareto_array = np.array([[p, t, *pareto_frontier_B_b(p, a, m, u, t)] for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] for t in np.linspace(1, 30, 30)])


    pareto_df = pd.DataFrame({'P': pareto_array[:, 0], 'T': pareto_array[:, 1], 'B': pareto_array[:, 2], 'Bid': pareto_array[:, 3]})
    pareto_df = pareto_df.dropna()
    
    for spend, reach in zip(cum_ad_spend_android, exponential_effectiveness(cum_ad_spend_android, *popt)):
        df = df.append({'Cumulative Spent': spend, 'Cumulative Reach': reach, 'Inferred': 'Fit'}, ignore_index = True)
    
    graphJSON = plot_px(df)
    graphJSON2 = plot_pareto(pareto_df)    

    return render_template('rapor.html', graphJSON=graphJSON, graphJSON2=graphJSON2, m=m, u=u)

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
        df['Cumulative Reach'] = df['Reach'].cumsum() / 500000 # Audience size nasil belirleniyor
        
        # Son n gunu kampanya_df ye ekle df['Amount spent (USD)'] df['Reach']            

        #kampanya_df = pd.DataFrame({'Day': int(day), 'B': float(b), 'T': int(t), 'Bid': float(bid), 'P': float(p), 'M': float(m), 'U':float(u)}, index=[0])
        kampanya_df = pd.DataFrame(columns=['Day', 'B', 'T', 'Bid', 'P', 'M', 'U', 'Reach'])
        df2 = df.copy()
        df2.sort_values(by='Day', ascending=False)
        df2 = df2.tail(ONLINE_LEARNING_N)
        for i in range(ONLINE_LEARNING_N):
            new_entry = pd.DataFrame({'Day': 0, 'B': df2['Amount spent (USD)'].iloc[i], 'T': 0, 'Bid': 0, 'P': 0, 'M': 0, 'U': 0, 'Reach': df2['Reach'].iloc[i]}, index=[0])
            kampanya_df = kampanya_df.append(new_entry, ignore_index=True)
        kampanya_df = kampanya_df.append(pd.DataFrame({'Day': int(day), 'B': float(b), 'T': int(t), 'Bid': float(bid), 'P': float(p), 'M': float(m), 'U':float(u), 'Reach':0},  index=[0]), ignore_index=True)
        kampanya_df.to_pickle("./kampanya_df.pickle") # Yeni kampanya için overwrite
             
    kampanya_df = pd.read_pickle("./kampanya_df.pickle")
    print(kampanya_df)

    if request.method == 'POST' and 'Dün Harcanılan Para' in request.form and 'Dün Alınan Sonuç' in request.form:
        
        total_effectiveness_arr = [] # Step 1
        #for j in range(n): # Son n tane sample icin icin g_star hesapla

            #total_effectiveness += exponential_effectiveness(kampanya_df.iloc[-1 * (1 + j)['Bid']], kampanya_df.iloc[-1]['M'], kampanya_df.iloc[-1]['U']) # Son n veriyi cek

        #g_star = g_16(WORD_OF_MOUTH, INITIAL_EXPOSURE, total_effectiveness)


        b = kampanya_df.iloc[-1] - request.form['Dün Harcanılan Para'] # Step 2
        
        
    elif request.method == 'POST':
        flash('Lütfen formu doldurun')

    return render_template('kampanya.html')
    
if __name__ == "__main__":
    app.run(debug=True)
    
    
