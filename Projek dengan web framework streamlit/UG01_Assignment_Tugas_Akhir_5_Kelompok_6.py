import streamlit as st
import pandas as pd
import requests
import numpy as np
import datetime
import json
# library untuk visualisasi
from plotly import graph_objs as go
# library untuk split data : testing data dan training data
from sklearn.model_selection import train_test_split
# library untuk model linear regression
from sklearn.linear_model import LinearRegression # linear regression
from sklearn.svm import SVR # support vector regression
from keras.models import Sequential # LSTM
from keras.layers import Activation, Dense, Dropout, LSTM # LSTM
# library untuk normalisasi
from sklearn.preprocessing import MinMaxScaler
# library untuk menghitung evaluasi model
from sklearn import metrics  
# api key untuk ekstrak data dari web alpha vantage
api_key = "2A3KMY0GNM9JUBKA"
# start date pengambilan dataset
START = "2015-01-01"

# HISTORICAL DATA function ~ TIME_SERIES_DAILY
@st.cache
def get_historical_data(symbol, start_date = None):
    api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}&outputsize=full'
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df[f'Time Series (Daily)']).T
    df = df.rename(columns = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', 
                                '5. adjusted close': 'adj close', '6. volume': 'volume'})
    for i in df.columns:
        df[i] = df[i].astype(float)
    df.index = pd.to_datetime(df.index)
    # delete kolom 'dividen amount' dan kolom 'split coefficient'
    df = df.iloc[::-1].drop(['adj close','7. dividend amount', '8. split coefficient'], axis = 1) 
    if start_date:
        df = df[df.index >= start_date]
    return df

# plot open and close price
def plot_data():
    fig = go.Figure()
    X = data.index
    fig.add_trace(go.Scatter(x=X, y=data['open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=X, y=data['close'], name="stock_close"))
    fig.layout.update(title_text='Data Visualization', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# plot prediction and actual
def pred_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['predict'], name="prediction"))
    fig.add_trace(go.Scatter(x=data['date'], y=data['actual price'], name="actual price"))
    fig.layout.update(title_text='prediction Data Visualization', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


page = st.sidebar.selectbox('Select page',['Homepage','Regression page','deep learning page'])

if page == 'Homepage':
    st.title('Projek prediksi harga saham menggunakan machine learning')

    st.subheader("Latar Belakang : ")
    text1 = """
    Salah satu pelajaran kepada para pelaku pasar saham seperti investor dan trader, yang cukup terkenal adalah “don’t try to time the market”. Pesan itu berasal dari Peter Lynch, seorang manajer investasi terkenal di wall street akan kesuksesannya dalam menggandakan uang kliennya dari senilai US$ 18 Juta menjadi US$ 14 Miliar selama 23 tahun dia berkarir. Selama beliau berkarir, Peter Lynch berpendapat bahwa merupakan hal yang mustahil untuk memprediksi harga saham dari suatu perusahaan. Karena sifat alami dari pasar saham yang dikenal dengan pergerakan harga yang dinamis, volatile dan non-linier. Prediksi harga saham yang akurat sangat menantang karena banyak faktor (makro dan mikro), seperti politik, kondisi ekonomi global, kejadian tak terduga, kinerja keuangan perusahaan, dan sebagainya.
    """
    text2 = """
    Tapi, semua ini juga berarti ada banyak data untuk menemukan polanya. Jadi, analis keuangan, peneliti, dan ilmuwan data terus mengeksplorasi teknik analitik untuk mendeteksi tren pasar saham. Yang dimana, terlepas dari volatilitas, harga saham bukan hanya angka yang dihasilkan secara acak. Harga saham dapat dianalisis sebagai urutan data time series. Prediksi data time series atau time series forecasting (memprediksi nilai masa depan berdasarkan nilai historis) merupakan salah satu eksperimen yang masih memiliki ruang eksplorasi yang cukup banyak sehingga mengundang para ilmuwan data untuk mencoba membuat algoritma yang bisa memprediksi dengan optimal data time series kedepannya, seperti prediksi harga saham suatu perusahaan. 
    """
    st.write(text1)
    st.write(text2)

    st.subheader("Deskripsi Projek :")
    desk1 = """
    - Projek merupakan sistem prediksi harga saham sesuai pilihan dari user.
    - Untuk dataset menggunakan API dari alpha vantage dengan mengambil dataset harga saham dari tanggal 1 januari 2015 (bisa diubah dalam code-nya)
    - Terdapat 3 page, homepage yang berisi latar belakang dan deksripsi projek, regression page yang berisi sistem prediksi harga saham dengan berdasarkan input harga opening dari user. Dan forecast page yang berisi prediksi harga saham di hari esok berdasarkan 5 harga opening di hari sebelumnya.
    - untuk model regresi, menggunakan algoritma linear regression dan support vector machine.
    - untuk model forecast, menggunakan algoritma Long Short Term Memory (LSTM).
    """
    st.write(desk1)

    st.subheader("source code : ")
    link1 = """
    - link laporan projek (google docs) : <------->
    - link youtube presentasi projek : <-------->
    - link github source code : <------------->
    """
    st.write(link1)


elif page == 'Regression page':
    mod = ("regresi linear", "support vector regression")
    selected_mod = st.sidebar.selectbox('Select model',mod)

    st.title("bereksperimen memprediksi harga saham dengan model regresi")

    reg1 = """
    Disini sistem akan memprediksi harga saham dengan berdasarkan data input dari user. Untuk ini kami menggunakan algoritma model regresi seperti model regresi linear, dan support vector regression. Data input kami menggunakan harga pembukaan saham untuk memprediksi harga penutupan di hari yang sama. Alasan dipilihnya harga pembukaan sebagai data masukan ke model karena nilai korelasi yang tinggi antara harga pembukaan dan harga penutupan pada suatu saham.
    """
    reg2 = """
    pertama user akan model regresi mana yang ingin digunakan, beserta saham yang ingin di prediksi. setelah itu user harus memasukkan harga pembukaan dari saham tersebut untuk dijadikan data masukan ke model yang dipilih. 
    """
    st.write(reg1)
    st.write(reg2)

    teks_lr = """
    Salah satu model regresi yang paling sederhana dalam pembelajaran mesin. Simple linear Regression hanya mempunyai 1 independent variable (x). Walaupun sederhana, algoritma ini merupakan salah satu algoritma yang sangat populer karena simple tapi powerful. Jadi, secara sederhana tujuan dari Simple Linear Regression adalah untuk memprediksi nilai dari y dengan mengetahui nilai x dan menemukan nilai m dan b yang errornya paling minimal.
    """
    teks_svr = """
    Analisis Support Vector Machines (SVM) adalah salah satu dari tool machine learning yang populer untuk klasifikasi dan regresi, mendukung regresi linier dan nonlinier yang dapat kita sebut sebagai SVR. SVR juga dapat digunakan dalam memprediksi data yang berbentuk time series, seperti harga saham dan harga emas. Keuntungan menggunakan SVM adalah SVM bekerja dengan baik ketika ada pemisahan margin yang jelas antar kelas, SVM lebih efektif dalam ruang dimensi tinggi (dalam kasus di mana jumlah dimensi lebih besar dari jumlah sampel), dan SVM menggunakan subset poin train dalam fungsi keputusan (disebut support vector), sehingga relatif hemat memori.
    """

    if selected_mod == "regresi linear":
        st.subheader("model linear regression")
        st.write(teks_lr)
    elif selected_mod == "support vector regression":
        st.subheader("model support vector regression")
        st.write(teks_svr)
    
    st.subheader("silahkan pilih saham yang ingin dianalisa :")
    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
    selected_stock = st.selectbox('saham :', stocks)

    data_load_state = st.text('Loading data...')
    data = get_historical_data(selected_stock, START)
    data_load_state.text('Loading data... done!')

    # show dataset
    if st.checkbox('show dataset (before normalization)'):
        st.subheader('dataset')
        st.write(data)
    # dataset di normalize dan di show
    scaler = MinMaxScaler()
    normalizeData = pd.DataFrame(scaler.fit_transform(data))
    normalizeData.columns = data.columns
    if st.checkbox('show dataset (after normalization)'):
        st.subheader('dataset')
        st.write("angka di lakukan normalisasi menggunakan min-max scaler sehingga menghasilkan nilai antara 0 - 1")
        st.write(normalizeData)
    # Plot open and close price 
    plot_data()   
    # plot volume
    st.line_chart(data['volume']) 

    st.subheader("Hasil prediksi model")

    if selected_mod == "regresi linear":
        # model linear regression
        x_lr = normalizeData["open"].values.reshape(-1,1)
        y_lr = normalizeData["close"].values.reshape(-1,1)

        X_train_temp ,X_test_temp ,y_train_temp, y_test_temp = train_test_split(x_lr, y_lr, train_size = 0.8, shuffle=False)
        date_train, date_test = train_test_split(data.index, train_size=0.8, shuffle=False)

        model_lr = LinearRegression()
        model_lr.fit(X_train_temp,y_train_temp)

        # prediction
        lr_predict = model_lr.predict(X_test_temp)

        result_lr = pd.DataFrame({"date": date_test, "predict" : lr_predict.ravel(), "actual price" : X_test_temp.ravel()})

        # show prediction result
        if st.checkbox('show prediction'): 
            st.subheader('prediction result')
            st.write(result_lr) # dataframe
            st.write("dari model diatas, evaluasinya sebesar :")
            st.write('Mean Absolute Error:', metrics.mean_absolute_error(result_lr['predict'], result_lr['actual price']))  
            st.write('Mean Squared Error:', metrics.mean_squared_error(result_lr['predict'], result_lr['actual price']))  
            st.write('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(result_lr['predict'], result_lr['actual price'])))   
        pred_data(result_lr) # chart

        # input data ~ open price
        open_input = st.text_input("input your stock open price")
        check = st.button("predict")
        if check:
            open_input = float(open_input)
            result = model_lr.predict(np.array([open_input]).reshape(-1,1))
            st.write("prediction of the close price")
            st.write(result)

    elif selected_mod == "support vector regression":
        x_svr = normalizeData["open"].values.reshape(-1,1)
        y_svr = normalizeData["close"].values.reshape(-1,1)
        X_train_svr, X_test_svr, y_train_svr, y_test_svr = train_test_split(x_svr, y_svr, train_size = 0.8, shuffle=False)
        date_train, date_test = train_test_split(data.index, train_size=0.8, shuffle=False)

        # Model & Train; Model SVR menggunakan nilai Epsilon 0.01 dan Gamma 0.01
        model_svr = SVR(kernel= 'rbf', epsilon=1e-2, gamma=1e-2)
        model_svr.fit(X_train_svr, y_train_svr)

        # Prediction
        svr_preds = model_svr.predict(X_test_svr)
        
        result_svr = pd.DataFrame({"date": date_test, "predict" : svr_preds.ravel(), "actual price" : X_test_svr.ravel()})
        # show prediction result
        if st.checkbox('show prediction'): 
            st.subheader('prediction result')
            st.write(result_svr) # dataframe
            st.write("dari model diatas, evaluasinya sebesar :")
            st.write('Mean Absolute Error:', metrics.mean_absolute_error(result_svr['predict'], result_svr['actual price']))  
            st.write('Mean Squared Error:', metrics.mean_squared_error(result_svr['predict'], result_svr['actual price']))  
            st.write('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(result_svr['predict'], result_svr['actual price'])))   
        pred_data(result_svr) # chart

        # input data ~ open price
        open_input = st.text_input("input your stock open price")
        check = st.button("predict")
        if check:
            open_input = float(open_input)
            result = model_svr.predict(np.array([open_input]).reshape(-1,1))
            st.write("prediction of the close price")
            st.write(result)

elif page == 'deep learning page':
    st.title("bereksperimen memprediksi harga saham dengan model Deep learning, LSTM")

    st.subheader("silahkan pilih saham yang ingin dianalisa :")
    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
    selected_stock = st.selectbox('saham :', stocks)

    data_load_state = st.text('Loading data...')
    data2 = get_historical_data(selected_stock, START)
    data_load_state.text('Loading data... done!')

    ds_close = data2.iloc[:,4:5].values

    scaler = MinMaxScaler(feature_range = (0, 1))
    scaled_ds_close = scaler.fit_transform(ds_close)

    step = 21

    X_train = []
    y_train = []

    for i in range(step, scaled_ds_close.shape[0]):              
        X_train.append(scaled_ds_close[i-step:i, 0])
        y_train.append(scaled_ds_close[i, 0])

    stockFeature = np.array(X_train[:-1])                   
    stockLabel = np.array(y_train[:-1])

    n = len(ds_close)
    split = int(n*0.8)

    X_train_lstm = stockFeature[:split]
    X_test_lstm = stockFeature[split:]
    y_train_lstm = stockLabel[:split]
    y_test_lstm = stockLabel[split:]

    X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
    X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

    #membangun model lstm
    model = Sequential()

    model.add(LSTM(units=40, return_sequences=True))
    model.add(Dropout(0.15))

    model.add(LSTM(units=40, return_sequences=True))
    model.add(Dropout(0.15))

    model.add(LSTM(units=40))
    model.add(Dropout(0.15))

    model.add(Dense(1))
    #prediksi dengan model lstm

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=100)

    preds = model.predict(X_test_lstm)

    lstm_predictions = scaler.inverse_transform(preds)
    y_test_lstm = scaler.inverse_transform(y_test_lstm.reshape(-1,1))

    st.write("prediction price :")
    st.line_chart(lstm_predictions)
    st.write("actual price")
    st.line_chart(y_test_lstm)
