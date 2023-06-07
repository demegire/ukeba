import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from flask import Flask, flash, render_template, request, redirect, url_for
from flask_ngrok import run_with_ngrok
from utils import exponential_effectiveness, new_bid, pareto_frontier_B_b, maximize_ltv1ltv2_sum, revenue_estimator
from multi import multi_pareto_frontier_B_b
from werkzeug.utils import secure_filename
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import json
import os

import plotly
import plotly.express as px

from constants import WORD_OF_MOUTH, INITIAL_EXPOSURE, ONLINE_LEARNING_N, AUDIENCE_SIZE, DECAY_RATE, P0, META_AUDIENCE_SIZE, IRONSOURCE_AUDIENCE_SIZE, IRONSOURCE_LTV, META_LTV

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'pdf', 'xlsx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"
#run_with_ngrok(app)

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
        if 'file' not in request.files and 'kitlefile' not in request.files and 'platformfile' not in request.files:
            return redirect(request.url)
        if 'file' in request.files.keys():
            file = request.files['file']
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
                    df = pd.read_excel(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    df = df.rename(columns={'Unnamed: 0': 'Date'})
                    df = df[df['Date'].notna()]
                    df['Cost'] = df['Cost'].str.replace('$', '').str.replace(',', '.') #1.000,23 seklinde . lar yok diye varsaydik
                    df['Cost'] = df['Cost'].astype(float)
                    df.to_pickle("./df.pickle")
                    return redirect('/rapor.html')

        if 'kitlefile' in request.files.keys():
            file = request.files['kitlefile']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                xl = pd.ExcelFile(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                for name in xl.sheet_names:
                    df = xl.parse(name)
                    if 'Unnamed: 0' in df.columns: # For Meta files
                        df = df.rename(columns={'Unnamed: 0': 'Date'})
                        df = df[df['Date'].notna()]
                        df['Cost'] = df['Cost'].str.replace('$', '').str.replace(',', '.') #1.000,23 seklinde . lar yok diye varsaydik
                    df['Cost'] = df['Cost'].astype(float)
                    df.to_pickle("./{}.pickle".format(name))
                return redirect('/rapor_kitle.html')        
            
        if 'platformfile' in request.files.keys():
            file = request.files['platformfile']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                xl = pd.ExcelFile(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                for name in xl.sheet_names:
                    df = xl.parse(name)
                    if 'Unnamed: 0' in df.columns: # For Meta files
                        df = df.rename(columns={'Unnamed: 0': 'Date'})
                        df = df[df['Date'].notna()]
                        df['Cost'] = df['Cost'].str.replace('$', '').str.replace(',', '.') #1.000,23 seklinde . lar yok diye varsaydik
                    df['Cost'] = df['Cost'].astype(float)
                    df.to_pickle("./{}.pickle".format(name))
                return redirect('/rapor_platform.html') 
            
    return render_template("homepage.html")
    
@app.route('/rapor.html', methods = ['GET', 'POST'])
def rapor():

    def plot_px(df):
        fig = px.line(df, x='Verilen İhale Değeri', y='Sonuç', markers=True, color='Veri Kaynağı', title="Verilen İhale Değeri vs Sonuç")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return graphJSON
    
    def plot_pareto(df):
        fig = px.line(df, x='B', y='T', markers=True, color='P', hover_data=['Bid', 'Profit'], title="Farklı Bütçe, Uzunluk ve İndirme Sayıları için Kampanya Senaryoları")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return graphJSON

    graphJSON = {}

    df = pd.read_pickle("./df.pickle")

    df = df.sort_values(by=['Cost'])
    df['Verilen İhale Değeri'] = df['Cost'] 
    df['Sonuç Yüzdesi'] = df['Install'] / AUDIENCE_SIZE
    df['Sonuç'] = df['Install']

    cum_ad_spend = np.array(df['Verilen İhale Değeri'], dtype='f')
    cum_result = np.array(df['Sonuç Yüzdesi'], dtype='f')
    df['Veri Kaynağı'] = 'Gerçek'
        
    popt, pcov = curve_fit(exponential_effectiveness, cum_ad_spend, cum_result, p0=P0, maxfev=5000)
    gecici_df = pd.read_excel(os.path.join(app.config['UPLOAD_FOLDER'], 'toyshop_dailybreakdown_cells.xlsx')) # Duzelt
    [m, u] = popt
    pareto_array = np.array([[p, t, *pareto_frontier_B_b(p, WORD_OF_MOUTH, m, u, t, INITIAL_EXPOSURE)] for p in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.02, 0.03, 0.04, 0.05] for t in np.linspace(1, 30, 30)])
    nan_rows = np.isnan(pareto_array).any(axis=1)
    pareto_array = pareto_array[~nan_rows]
    profits = np.array([revenue_estimator(gecici_df, p, AUDIENCE_SIZE, int(t), B, b) for p, t, B, b in pareto_array])
    pareto_df = pd.DataFrame({'P': (AUDIENCE_SIZE * pareto_array[:, 0]).astype(int), 'T': pareto_array[:, 1], 'B': np.around(pareto_array[:, 2], decimals=2), 'Bid': np.around(pareto_array[:, 3], decimals=2), 'Profit': np.around(profits, decimals=2), 'ROI': np.around((profits / pareto_array[:, 2]), decimals=2)})
    pareto_df = pareto_df.dropna()
    print(pareto_df.idxmax())


    for spend, reach in zip(cum_ad_spend, exponential_effectiveness(cum_ad_spend, *popt)):
        df = df.append({'Verilen İhale Değeri': spend, 'Sonuç': AUDIENCE_SIZE * reach, 'Veri Kaynağı': 'Tahmin'}, ignore_index = True)

    ss_res = np.sum((cum_result - exponential_effectiveness(cum_ad_spend, *popt)) ** 2)
    ss_tot = np.sum((cum_result - np.mean(cum_result)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    graphJSON = plot_px(df)
    graphJSON2 = plot_pareto(pareto_df)    

    return render_template('rapor.html', graphJSON=graphJSON, graphJSON2=graphJSON2, m=m, u=u, kitle='Etkililik Fonksiyonu R^2 Değeri: ' + "{:.2f}".format(r2))

@app.route('/rapor_platform.html', methods =['GET', 'POST'])
def rapor_platform():
    budgetJSON = ''
    graphJSON1 = ''
    graphJSON2 = ''
    graphJSON3 = ''
    if request.method == 'POST' and request.form['Bütçe'] and request.form['Ironsource Gelir Çarpanı'] and request.form['Meta Gelir Çarpanı']:
        df1 = pd.read_pickle("./Meta.pickle") 
        
        df1 = df1.sort_values(by=['Cost'])
        df1['Verilen İhale Değeri'] = df1['Cost']
        df1['Sonuç Yüzdesi'] = df1['Install'] / AUDIENCE_SIZE

        cum_ad_spend_1 = np.array(df1['Verilen İhale Değeri'], dtype='f')
        cum_result_1 = np.array(df1['Sonuç Yüzdesi'], dtype='f')
            
        popt_1, _ = curve_fit(exponential_effectiveness, cum_ad_spend_1, cum_result_1, p0=P0, maxfev=5000)

        df2 = pd.read_pickle("./Ironsource.pickle") 
        
        df2 = df2.sort_values(by=['Date'])
        df2['Verilen İhale Değeri'] = df2['Cost']
        df2['Sonuç Yüzdesi'] = df2['Install']  / AUDIENCE_SIZE

        cum_ad_spend_2 = np.array(df2['Verilen İhale Değeri'], dtype='f')
        cum_result_2 = np.array(df2['Sonuç Yüzdesi'], dtype='f')
            
        popt_2, _ = curve_fit(exponential_effectiveness, cum_ad_spend_2, cum_result_2, p0=P0, maxfev=5000)

        b_limit = float(request.form['Bütçe'])
        platform_pars_1 = [WORD_OF_MOUTH, META_AUDIENCE_SIZE, float(request.form['Meta Gelir Çarpanı'])] # a,n,ltv - take as input
        m_u_1 = popt_1
        platform_pars_2 = [WORD_OF_MOUTH, IRONSOURCE_AUDIENCE_SIZE, float(request.form['Ironsource Gelir Çarpanı'])]
        m_u_2 = popt_2
        
        ltv_results = maximize_ltv1ltv2_sum(platform_pars_1, m_u_1, platform_pars_2, m_u_2, b_limit=b_limit)    
        
        p1, p2, h1, h2, ltv1, ltv2, B1, B2, ltv_total = ltv_results
        
        data = {
        'budget_allocated': [B1, B2],
        'exposure_percentage': [p1, p2],
        'exposed_population': [h1, h2],
        'ltvs': [ltv1, ltv2],
        'names': ['Meta', 'Ironsource']
        }

        df = pd.DataFrame(data)
        
        fig = px.pie(df, values='budget_allocated', names='names', title='Allocated budgets')
            
        budgetJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Bar for exposure percentage comparison
        fig = px.bar(df, x='exposure_percentage', y='names', color='names', title='Exposed Percentage Comparison')
        graphJSON1 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        # Bar for exposure percentage comparison
        fig = px.bar(df, x='exposed_population', y='names', color='names', title='Exposed Population Comparison')
        graphJSON2 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Bar for estimated LTV comparison
        fig = px.bar(df, x='ltvs', y='names', color='names', title='Estimated LTV Comparison')
        graphJSON3 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('rapor_platform.html', budgetJSON=budgetJSON, graphJSON1=graphJSON1, graphJSON2=graphJSON2, graphJSON3=graphJSON3)

@app.route('/rapor_kitle.html', methods =['GET', 'POST'])
def rapor_kitle():

    def plot_px(df, platform):
        fig = px.line(df, x='Verilen İhale Değeri', y='Sonuç', markers=True, color='Veri Kaynağı', title="{} için Verilen İhale Değeri vs Sonuç".format(platform))
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return graphJSON

    def plot_pareto(df):
        fig = px.line(df, x='B', y='T', color='P', markers=True, hover_data=['Bid Meta', 'Bid Ironsource', 'P Meta', 'P Ironsource'], title="Farklı Bütçe, Uzunluk ve İndirme Sayıları için Kampanya Senaryoları")
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return graphJSON
    
    kitle = ''
           
    df1 = pd.read_pickle("./Meta.pickle") 
    
    df1 = df1.sort_values(by=['Cost'])
    df1['Verilen İhale Değeri'] = df1['Cost']
    df1['Sonuç Yüzdesi'] = df1['Install'] / AUDIENCE_SIZE
    df1['Sonuç'] = df1['Install']
    df1['Veri Kaynağı'] = 'Gerçek'

    cum_ad_spend_1 = np.array(df1['Verilen İhale Değeri'], dtype='f')
    cum_result_1 = np.array(df1['Sonuç Yüzdesi'], dtype='f')
        
    popt_1, _ = curve_fit(exponential_effectiveness, cum_ad_spend_1, cum_result_1, p0=P0, maxfev=5000)
    for spend, reach in zip(cum_ad_spend_1, exponential_effectiveness(cum_ad_spend_1, *popt_1)):
        df1 = df1.append({'Verilen İhale Değeri': spend, 'Sonuç': AUDIENCE_SIZE * reach, 'Veri Kaynağı': 'Tahmin'}, ignore_index = True)
    effJSON1 = plot_px(df1, 'Meta')
    ss_res = np.sum((cum_result_1 - exponential_effectiveness(cum_ad_spend_1, *popt_1)) ** 2)
    ss_tot = np.sum((cum_result_1 - np.mean(cum_result_1)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    kitle += ('Meta Etkililik Fonksiyonu R^2 Değeri: ' + "{:.2f}".format(r2))
 
    df2 = pd.read_pickle("./Ironsource.pickle") 
    
    df2 = df2.sort_values(by=['Cost'])
    df2['Verilen İhale Değeri'] = df2['Cost']
    df2['Sonuç Yüzdesi'] = df2['Install']  / AUDIENCE_SIZE
    df2['Veri Kaynağı'] = 'Gerçek'
    df2['Sonuç'] = df2['Install']

    cum_ad_spend_2 = np.array(df2['Verilen İhale Değeri'], dtype='f')
    cum_result_2 = np.array(df2['Sonuç Yüzdesi'], dtype='f')
        
    popt_2, _ = curve_fit(exponential_effectiveness, cum_ad_spend_2, cum_result_2, p0=P0, maxfev=5000)

    for spend, reach in zip(cum_ad_spend_2, exponential_effectiveness(cum_ad_spend_2, *popt_2)):
        df2 = df2.append({'Verilen İhale Değeri': spend, 'Sonuç': AUDIENCE_SIZE * reach, 'Veri Kaynağı': 'Tahmin'}, ignore_index = True)
    effJSON2 = plot_px(df2, 'Ironsource')
    ss_res = np.sum((cum_result_2 - exponential_effectiveness(cum_ad_spend_2, *popt_2)) ** 2)
    ss_tot = np.sum((cum_result_2 - np.mean(cum_result_2)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    kitle += ('\n\nIronsource Etkililik Fonksiyonu R^2 Değeri: ' + "{:.2f}".format(r2))

    p_vals = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
    p_arr = [[p1, p2] for p1 in p_vals for p2 in p_vals]
    m = [popt_1[0], popt_2[0]]
    u = [popt_1[1], popt_2[1]]
    a = [WORD_OF_MOUTH, WORD_OF_MOUTH]
    q = [INITIAL_EXPOSURE, INITIAL_EXPOSURE]

    pareto_array = np.array([[*p, t, *multi_pareto_frontier_B_b(p, a, m, u, t, q)] for p in p_arr for t in np.linspace(1, 30, 30)])
    nan_rows = np.isnan(pareto_array).any(axis=1)
    pareto_array = pareto_array[~nan_rows]
    # Revenue ekle
    combined_p = META_AUDIENCE_SIZE * pareto_array[:, 0] + IRONSOURCE_AUDIENCE_SIZE * pareto_array[:, 1]
    pareto_df = pd.DataFrame({'P': combined_p.astype(int), 'T': pareto_array[:, 2], 'B': np.around(pareto_array[:, 3], decimals=2), 'Bid Meta': np.around(pareto_array[:, 4], decimals=2), 'Bid Ironsource': np.around(pareto_array[:, 5], decimals=2), 'P Meta': pareto_array[:, 0], 'P Ironsource': pareto_array[:, 1]})
    pareto_df = pareto_df.dropna()
    graphJSON2 = plot_pareto(pareto_df) 

    return render_template('rapor_kitle.html', graphJSON2=graphJSON2, effJSON1=effJSON1, effJSON2=effJSON2, kitle=kitle, m=m, u=u)

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
         
        kampanya_df = pd.DataFrame(columns=['Day', 'B', 'T', 'Bid', 'P', 'M', 'U', 'Reach'])
        df2 = df.copy()
        for i in range(len(df)):
            new_entry = pd.DataFrame({'Date': df2['Date'].iloc[i], 'Day': 0, 'B': 0, 'T': 0, 'Bid': df2['Cost'].iloc[i], 'P': 0, 'M': 0, 'U': 0, 'Reach':  df2['Install'].iloc[i] / AUDIENCE_SIZE}, index=[0])            
            kampanya_df = kampanya_df.append(new_entry, ignore_index=True)
        kampanya_df = kampanya_df.append(pd.DataFrame({'Day': int(day), 'B': float(b), 'T': int(t), 'Bid': float(bid), 'P': float(p) / AUDIENCE_SIZE, 'M': float(m), 'U':float(u), 'Reach':0},  index=[0]), ignore_index=True)
        kampanya_df.to_pickle("./kampanya_df.pickle")

        flash('Kalan Toplam Bütçe: ' + "{:.2f}".format(kampanya_df.iloc[-1]['B']))
             
    kampanya_df = pd.read_pickle("./kampanya_df.pickle")
    
    yeni_bid = 0

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
        #flash('Yeni günlük bütçe: ' + "{:.2f}".format(yeni_bid))
        if yeni_bid < 0:
             flash('Amaca ulaşıldı!')

        yeni_b = kampanya_df.iloc[-1]['B'] - harcanilan_para # Step 2
        if yeni_b < 0:
            flash('Bütçe aşıldı!')

        if yeni_bid > yeni_b:
            flash('Yeni günlük bütçe, kalan bütçeyi aşıyor!')

        if kampanya_df.iloc[-1]['T'] == 1:
            flash('Kampanya süresi doldu!')

        kampanya_df = kampanya_df.append(pd.DataFrame({'Day': kampanya_df.iloc[-1]['Day'] + 1, 'B': yeni_b, 'T':  kampanya_df.iloc[-1]['T']-1, 'Bid': yeni_bid, 'P':  kampanya_df.iloc[-1]['P'], 'M': m, 'U': u, 'Reach': kampanya_df.iloc[-1]['Reach'] + (sonuc / AUDIENCE_SIZE)}, index=[0]), ignore_index=True)        
        
        flash('Kalan Toplam Bütçe: ' + "{:.2f}".format(kampanya_df.iloc[-1]['B']))
        
        kampanya_df.to_pickle("./kampanya_df.pickle")

        print(kampanya_df)
        
    elif request.method == 'POST':
        flash('Lütfen formu doldurun')

    flash('Kalan hedef: ' + str(int((kampanya_df.iloc[-1]['P'] - kampanya_df.iloc[-1]['Reach']) * META_AUDIENCE_SIZE)))
    table_data = [[i[1]['Day'], int(i[1]['Reach'] * META_AUDIENCE_SIZE), round(i[1]['Bid'], 2)] for i in kampanya_df.tail(kampanya_df.iloc[-1]['Day']).iterrows()]

    return render_template('kampanya.html', yeni_bid=yeni_bid, table_data=table_data)

@app.route('/kampanya_platform.html', methods =['GET', 'POST'])
def kampanya_platform():
    return

@app.route('/kampanya_kitle.html', methods =['GET', 'POST'])
def kampanya_kitle():
    day = request.args.get('day')
    t = request.args.get('t')
    b = request.args.get('b')
    bid_kitle1 = request.args.get('bid_meta')
    bid_kitle2 = request.args.get('bid_iron')
    p_kitle1 = request.args.get('p_meta')
    p_kitle2 = request.args.get('p_iron')
    m = request.args.get('m')
    u = request.args.get('u')

    audiences = ['Kitle1', 'Kitle2']
    audience_dict = {'Kitle1': META_AUDIENCE_SIZE, 'Kitle2': IRONSOURCE_AUDIENCE_SIZE}
    bid_dict = {'Kitle1': bid_kitle1, 'Kitle2': bid_kitle2}
    p_dict = {'Kitle1': p_kitle1, 'Kitle2': p_kitle2}
    file_dict = {'Kitle1': 'Meta', 'Kitle2': 'Ironsource'}

    if day == "1":
        m = m.split(',')
        u = u.split(',')
        for idx, i in enumerate(audiences):
            df = pd.read_pickle('./{}.pickle'.format(file_dict[i]))
            flash('{} Eski Günlük Bütçe: '.format(i) + str(bid_dict[i]))
            
            kampanya_df = pd.DataFrame(columns=['Day', 'B', 'T', 'Bid', 'P', 'M', 'U', 'Reach'])
            df2 = df.copy()
            for r in range(len(df)):
                new_entry = pd.DataFrame({'Date': df2['Date'].iloc[r], 'Day': 0, 'B': 0, 'T': 0, 'Bid': df2['Cost'].iloc[r], 'P': 0, 'M': 0, 'U': 0, 'Reach':  df2['Install'].iloc[r] / audience_dict[i]}, index=[0])            
                kampanya_df = kampanya_df.append(new_entry, ignore_index=True)
            kampanya_df = kampanya_df.append(pd.DataFrame({'Day': int(day), 'B': float(b), 'T': int(t), 'Bid': float(bid_dict[i]), 'P': float(p_dict[i]), 'M': float(m[idx]), 'U':float(u[idx]), 'Reach':0},  index=[0]), ignore_index=True)
            kampanya_df.to_pickle("./kampanya_{}.pickle".format(i))

        flash('Kampanya Günü: ' + str(kampanya_df.iloc[-1]['T']))
        flash('Kalan Toplam Bütçe: ' + str(kampanya_df.iloc[-1]['B']))

    campaign_dfs = [pd.read_pickle("./kampanya_{}.pickle".format(i)) for i in audiences]
    new_bids = [0, 0]

    if request.method == 'POST' and request.form['Kitle 1 Dün Harcanılan Para'] and request.form['Kitle 1 Dün Alınan Sonuç'] and request.form['Kitle 2 Dün Harcanılan Para'] and request.form['Kitle 2 Dün Alınan Sonuç']:
        
        actual_expenditures = [float(request.form['Kitle 1 Dün Harcanılan Para']), float(request.form['Kitle 2 Dün Harcanılan Para'])] 
        results = [float(request.form['Kitle 1 Dün Alınan Sonuç']), float(request.form['Kitle 2 Dün Alınan Sonuç'])] 

        new_entries = [pd.DataFrame({'Day': campaign_dfs[idx].iloc[-1]['Day'] + 1, 'B': 0, 'T': 0, 'Bid': actual_expenditures[idx], 'P': 0, 'M': 0, 'U': 0, 'Reach':  results[idx] / audience_dict[aud]}, index=[0]) for idx, aud in enumerate(audiences)]
        
        campaign_dfs = [i.sort_values(by=['Date']) for i in campaign_dfs]
        temp_dfs = [campaign_dfs[i].append(new_entries[i], ignore_index=True) for i in range(len(audiences))]

        new_budget = campaign_dfs[0].iloc[-1]['B'] - sum(actual_expenditures) # Step 2

        if new_budget < 0:
            flash('Bütçe aşıldı!') # Out of money

        if campaign_dfs[0].iloc[-1]['T'] == 1:
            flash('Kampanya süresi doldu!') # Out of time

        for idx, df in enumerate(temp_dfs):
            df['Weights'] = np.array([np.power(DECAY_RATE, i) for i in range(len(df))]) # Exponential weighting, n>4000 underflow issues

            df['Verilen İhale Değeri'] = df['Bid']
            df['Sonuç Yüzdesi'] = df['Reach']

            df = df.sort_values(by=['Bid'])
            cum_ad_spend = np.array(df['Verilen İhale Değeri'], dtype='f')
            cum_result = np.array(df['Sonuç Yüzdesi'], dtype='f')
            
            p0 = [campaign_dfs[idx].iloc[-1]['M'], campaign_dfs[idx].iloc[-1]['U']]

            popt, _ = curve_fit(exponential_effectiveness, cum_ad_spend, cum_result, p0=p0, maxfev=5000, sigma=df['Weights'].to_numpy())
            [m, u] = popt

            new_bids[idx] = new_bid(campaign_dfs[idx].iloc[-1]['P'], WORD_OF_MOUTH, m, u, campaign_dfs[idx].iloc[-1]['T'], campaign_dfs[idx].iloc[-1]['Reach'])
            flash('{} Yeni bid: '.format(audiences[idx]) + "{:.2f}".format(new_bids[idx]))

            campaign_dfs[idx] = campaign_dfs[idx].append(pd.DataFrame({'Day': campaign_dfs[idx].iloc[-1]['Day'] + 1, 'B': new_budget, 'T':  campaign_dfs[idx].iloc[-1]['T']-1, 'Bid': new_bids[idx], 'P':  campaign_dfs[idx].iloc[-1]['P'], 'M': m, 'U': u, 'Reach': campaign_dfs[idx].iloc[-1]['Reach'] + (results[idx] / audience_dict[audiences[idx]])}, index=[0]), ignore_index=True)        
            campaign_dfs[idx].to_pickle("./kampanya_{}.pickle".format(audiences[idx]))
            print(campaign_dfs[idx])
        flash('Kampanya Günü: ' + str(campaign_dfs[0].iloc[-1]['T']))
        flash('Kalan Toplam Bütçe: ' + "{:.2f}".format(campaign_dfs[0].iloc[-1]['B']))
        #flash('Sonuç: ' + str(sum(df.iloc[-1]['Reach'] * AUDIENCE_SIZE))
        
    elif request.method == 'POST':
        flash('Lütfen formu doldurun')

    return render_template('kampanya_kitle.html', kitle1_bid=new_bids[0], kitle2_bid=new_bids[1])
    
if __name__ == "__main__":
    app.run(debug=True)
    
    
