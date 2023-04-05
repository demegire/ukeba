from flask import Flask, flash, render_template, request, redirect, url_for, session, send_file
from werkzeug.utils import secure_filename
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
import plotly.figure_factory as ff
from io import BytesIO
from snap7 import util
import pandas as pd
import numpy as np
import requests
import base64
import random
import snap7
import flask
import json
import os

import plotly
import plotly.express as px

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
        
    def func(x, m, u):
        return m * (1 - np.exp(-1 * u * x))

    def pareto_frontier_B_b(p, a, m, u, T):
        tmin = np.log((a * p + 1) / (1 - p)) / (m * (1 + a))
        B = ((-T/u) * np.log(1 - (tmin/T)))

        beta = u * (m - 1/((1 + a) * T) * np.log((a * p + 1)/((1 - p))))
        bid = 1 / u * np.log(m * u / beta)

        return B, bid

    graphJSON = {}

    #filename = session['messages']
    df = pd.read_pickle("./df.pickle")
    
    df = df.dropna()
    df = df[df['Campaign name'].str.contains('Android')]
    df = df.sort_values(by=['Day'])
    df['Cumulative Spent'] = df['Amount spent (USD)'].cumsum()
    df['Cumulative Reach'] = df['Reach'].cumsum() / 500000
    cum_ad_spend_android = np.array(df['Cumulative Spent'])
    cum_reach_android = np.array(df['Cumulative Reach'])
    df['Inferred'] = 'Ground Truth'
        
    popt, pcov = curve_fit(func, cum_ad_spend_android, cum_reach_android)

    [m, u] = popt
    a = 2
    pareto_array = np.array([[p, t, *pareto_frontier_B_b(p, a, m, u, t)] for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] for t in np.linspace(1, 30, 30)])


    pareto_df = pd.DataFrame({'P': pareto_array[:, 0], 'T': pareto_array[:, 1], 'B': pareto_array[:, 2], 'Bid': pareto_array[:, 3]})
    pareto_df = pareto_df.dropna()
    
    for spend, reach in zip(cum_ad_spend_android, func(cum_ad_spend_android, *popt)):
        df = df.append({'Cumulative Spent': spend, 'Cumulative Reach': reach, 'Inferred': 'Fit'}, ignore_index = True)
    
    graphJSON = plot_px(df)
    graphJSON2 = plot_pareto(pareto_df)    
    
    if request.method == 'POST' and 'beginningdate' in request.form and 'enddate' in request.form and 'Malzeme' in request.form:
        if request.form['beginningdate'] and request.form['enddate']:
            df['Malzeme Kodu'] = [int(float(x)) for x in df['Malzeme Kodu']]                
            malz1 = df[df['Malzeme Kodu'] == int(request.form['Malzeme'])]
            
            beginningdate = pd.to_datetime(request.form['beginningdate'])
            enddate = pd.to_datetime(request.form['enddate'])
            malz1 = malz1[malz1['Kayıdın Eklenme Zamanı'] > beginningdate]
            malz1 = malz1[malz1['Kayıdın Eklenme Zamanı'] < enddate]
                                  
            malz1 = malz1[malz1['Ölçülen Ağırlık'] < malz1['Ölçülen Ağırlık'].median() * 1.2] #ekstrem deger filtreleme
            malz1 = malz1[malz1['Ölçülen Ağırlık'] > malz1['Ölçülen Ağırlık'].median() * 0.8]
            
            graphJSON, graphJSON2, graphJSON3 = plot_px(malz1)
            flash('Seçilen malzeme: ' + str(int(request.form['Malzeme'])) + '\n')
            if len(pd.unique(malz1['Ağırlık Alt Limit'])) > 1:
                flash('Birden fazla limit tespit edildi! En son değer kullanılıyor.')
        else:        
            df['Malzeme Kodu'] = [int(float(x)) for x in df['Malzeme Kodu']]                      
            malz1 = df[df['Malzeme Kodu'] == int(request.form['Malzeme'])]
                                
            malz1 = malz1[malz1['Ölçülen Ağırlık'] < malz1['Ölçülen Ağırlık'].median() * 1.2] #ekstrem deger filtreleme
            malz1 = malz1[malz1['Ölçülen Ağırlık'] > malz1['Ölçülen Ağırlık'].median() * 0.8]
            
            graphJSON, graphJSON2, graphJSON3 = plot_px(malz1)
            flash('Seçilen malzeme: ' + str(int(request.form['Malzeme'])) + '\n')
            if len(pd.unique(malz1['Ağırlık Alt Limit'])) > 1:
                flash('Birden fazla limit tespit edildi! En son değer kullanılıyor.')
                
    elif request.method == 'POST':
        flash('Lütfen formu doldurun')
    
    return render_template('rapor.html', graphJSON=graphJSON, graphJSON2=graphJSON2)
    
if __name__ == "__main__":
    app.run(debug=True)
    
    
