import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
 
import pandas as pd
import numpy as np
 
import matplotlib.pyplot as plt
 
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
 
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

 
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
import glob

import warnings
warnings.filterwarnings('ignore')

app = dash.Dash()
server = app.server

# join all output files
folder_path = './output'

csv_files = glob.glob(folder_path + '/*.csv')
dfs = []

for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenate the individual DataFrames into a single DataFrame
df = pd.concat(dfs, ignore_index=True)
print(df)
 
app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='NSE-TATAGLOBAL Stock Data',children=[
            dcc.Graph(id='running')
        ]),

        dcc.Tab(label='Old Prediction Stock Data', children=[
            html.Div([
                html.H1("Stocks Actual vs Predictions by only RF", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'STB', 'value': 'STB'},
                                      {'label': 'HAG','value': 'HAG'}, 
                                      {'label': 'Facebook', 'value': 'FB'}, 
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['STB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='none'),
                html.H1("Stocks Actual vs Predictions by RF-MV", style={'textAlign': 'center'}),
                dcc.Graph(id='mv'),
                html.H1("Stocks Actual vs Predictions by RF-Opt", style={'textAlign': 'center'}),
                dcc.Graph(id='opt'),
                html.H1("Stocks Actual vs Predictions by RF-MV-Opt", style={'textAlign': 'center'}),
                dcc.Graph(id='mv-opt'),
            ], className="container"),
        ])
    ])
])
 
 
 
@app.callback(Output('running', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    data = pd.read_csv("./Dataset/STB.csv")
    data.rename(columns={"trunc_time":"Date","open_price":"open","high_price":"high","low_price":"low","close_price":"Close"}, inplace= True)

    from sklearn.preprocessing import MinMaxScaler
    closedf = data[['Date','Close']]
    print(closedf)
    print("Shape of close dataframe:", closedf.shape)
    close_stock = closedf.copy()
    del closedf['Date']
    scaler=MinMaxScaler(feature_range=(0,1))
    closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
    print(closedf.shape)
    time_step = 20

    training_size=int(len(closedf)*0.8)
    test_size=len(closedf)-training_size
    train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
    print("train_data: ", train_data.shape)
    print("test_data: ", test_data.shape)
    train_data

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)


    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 10
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    print(test_data.shape)
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    print("X_test: ", X_test.shape)
    print("y_test", y_test.shape)

    from sklearn.model_selection import KFold
    import time
    import psutil

    # cross validation
    # evaluate performance of model, can be used to be the target function in optuna
    # ref: from sklearn.model_selection import cross_val_score
    def cross_val_score(estimator, X, y, cv=5, scoring=None):
        scores = []
        
        # scoring: target function, if not provided it will be r2
        if scoring is None:
            scoring = r2_score
        
        cv_splitter = KFold(n_splits=cv, shuffle=True)
        
        for train_indices, test_indices in cv_splitter.split(X):
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            
            score = scoring(y_test, y_pred)
            scores.append(score)
        
        return np.mean(scores) # return average score of `cv` times run


    def r2_score(y_true, y_pred):
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (numerator / denominator)
        return r2

    def measure_system_metrics():
        cpu_percent = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        network_io = psutil.net_io_counters()
        
        return cpu_percent, memory_usage, disk_usage, network_io

    from sklearn.metrics import mean_squared_error
    import optuna

    # Define the objective function for Optuna
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        max_features = trial.suggest_int("max_features", 1, X_train.shape[1])


        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )

        scores = cross_val_score(model, X_train, y_train, cv=5)

        return scores

    # Use Optuna to optimize hyperparameters
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=1)
    best_params = study.best_params

    rf = RandomForestRegressor(**best_params)

    start_time = time.time()
    cpu_percent, memory_usage, disk_usage, network_io = measure_system_metrics()

    print(f"CPU usage: {cpu_percent}%")
    print(f"Memory usage: {memory_usage}%")
    print(f"Disk usage: {disk_usage}%")
    print(f"Network I/O: {network_io}")

    scores = cross_val_score(rf, X_train, y_train, cv=5)
    print("Cross-validated my model r2:", scores)
    end_time = time.time()

    # Measure system metrics after code execution
    cpu_percent, memory_usage, disk_usage, network_io = measure_system_metrics()

    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")
    print(f"CPU usage: {cpu_percent}%")
    print(f"Memory usage: {memory_usage}%")
    print(f"Disk usage: {disk_usage}%")
    print(f"Network I/O: {network_io}")

    rf.fit(X_train, y_train)

    # Lets Do the prediction 

    RF_train_predict=rf.predict(X_train)
    RF_test_predict=rf.predict(X_test)
    # print("Train data prediction:", train_predict)
    # # print("Test data prediction:", test_predict)
    RF_train_predict = RF_train_predict.reshape(-1,1)
    RF_test_predict = RF_test_predict.reshape(-1,1)

    RF_train_predict = scaler.inverse_transform(RF_train_predict)
    RF_test_predict = scaler.inverse_transform(RF_test_predict)

    look_back=time_step
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(RF_train_predict)+(look_back*2)+1:len(closedf)-1, :] = RF_test_predict
    close_stock['Predictions']=testPredictPlot
    print(close_stock['Close'])
    print(close_stock['Predictions'])
    trace1 = []
    trace2 = []
    trace1.append(
        go.Scatter(x=close_stock['Date'],
                    y=close_stock['Close'],
                    mode='lines', opacity=0.7, 
                    name=f'Actual STB',textposition='bottom center'))
    trace2.append(
        go.Scatter(x=close_stock['Date'],
                    y=close_stock['Predictions'],
                    mode='lines', opacity=0.6,
                    name=f'Prediction STB',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for  Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure
 
 
 
@app.callback(Output('none', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"STB": "STB","HAG": "HAG","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    trace3 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[(df["Stock"] == stock) & (df["Model"] == "SkLearn") & (df["Method"] == "No")]["Date"],
                     y=df[(df["Stock"] == stock) & (df["Model"] == "SkLearn") & (df["Method"] == "No")]["Close"],
                     mode='lines', opacity=0.7, 
                     name=f'Actual {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[(df["Stock"] == stock) & (df["Model"] == "SkLearn") & (df["Method"] == "No")]["Date"],
                     y=df[(df["Stock"] == stock) & (df["Model"] == "SkLearn") & (df["Method"] == "No")]["Predictions"],
                     mode='lines', opacity=0.6,
                     name=f'SkLearn model Prediction {dropdown[stock]}',textposition='bottom center'))
        trace3.append(
          go.Scatter(x=df[(df["Stock"] == stock) & (df["Model"] == "DIY") & (df["Method"] == "No")]["Date"],
                     y=df[(df["Stock"] == stock) & (df["Model"] == "DIY") & (df["Method"] == "No")]["Predictions"],
                     mode='lines', opacity=0.6,
                     name=f'DIY model Prediction {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2, trace3]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Actual and Predictions Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1 Month', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6 Month', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (VND)"})}
    return figure

@app.callback(Output('mv', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"STB": "STB","HAG": "HAG","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    trace3 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[(df["Stock"] == stock) & (df["Model"] == "SkLearn") & (df["Method"] == "MV")]["trunc_time"],
                     y=df[(df["Stock"] == stock) & (df["Model"] == "SkLearn") & (df["Method"] == "MV")]["close_price"],
                     mode='lines', opacity=0.7, 
                     name=f'Actual {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[(df["Stock"] == stock) & (df["Model"] == "SkLearn") & (df["Method"] == "MV")]["trunc_time"],
                     y=df[(df["Stock"] == stock) & (df["Model"] == "SkLearn") & (df["Method"] == "MV")]["Predictions"],
                     mode='lines', opacity=0.6,
                     name=f'SkLearn model with MV Prediction {dropdown[stock]}',textposition='bottom center'))
        trace3.append(
          go.Scatter(x=df[(df["Stock"] == stock) & (df["Model"] == "DIY") & (df["Method"] == "MV")]["trunc_time"],
                     y=df[(df["Stock"] == stock) & (df["Model"] == "DIY") & (df["Method"] == "MV")]["Predictions"],
                     mode='lines', opacity=0.6,
                     name=f'DIY model with MV Prediction {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2, trace3]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Actual and Predictions Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1 Month', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6 Month', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (VND)"})}
    return figure

@app.callback(Output('opt', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"STB": "STB","HAG": "HAG","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    trace3 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[(df["Stock"] == stock) & (df["Model"] == "SkLearn") & (df["Method"] == "Opt")]["Date"],
                     y=df[(df["Stock"] == stock) & (df["Model"] == "SkLearn") & (df["Method"] == "Opt")]["Close"],
                     mode='lines', opacity=0.7, 
                     name=f'Actual {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[(df["Stock"] == stock) & (df["Model"] == "SkLearn") & (df["Method"] == "Opt")]["Date"],
                     y=df[(df["Stock"] == stock) & (df["Model"] == "SkLearn") & (df["Method"] == "Opt")]["Predictions"],
                     mode='lines', opacity=0.6,
                     name=f'SkLearn model with Opt Prediction {dropdown[stock]}',textposition='bottom center'))
        trace3.append(
          go.Scatter(x=df[(df["Stock"] == stock) & (df["Model"] == "DIY") & (df["Method"] == "Opt")]["Date"],
                     y=df[(df["Stock"] == stock) & (df["Model"] == "DIY") & (df["Method"] == "Opt")]["Predictions"],
                     mode='lines', opacity=0.6,
                     name=f'DIY model with Opt Prediction {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2, trace3]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Actual and Predictions Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1 Month', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6 Month', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (VND)"})}
    return figure

@app.callback(Output('mv-opt', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"STB": "STB","HAG": "HAG","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    trace3 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[(df["Stock"] == stock) & (df["Model"] == "SkLearn") & (df["Method"] == "MV_Opt")]["trunc_time"],
                     y=df[(df["Stock"] == stock) & (df["Model"] == "SkLearn") & (df["Method"] == "MV_Opt")]["close_price"],
                     mode='lines', opacity=0.7, 
                     name=f'Actual {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[(df["Stock"] == stock) & (df["Model"] == "SkLearn") & (df["Method"] == "MV_Opt")]["trunc_time"],
                     y=df[(df["Stock"] == stock) & (df["Model"] == "SkLearn") & (df["Method"] == "MV_Opt")]["Predictions"],
                     mode='lines', opacity=0.6,
                     name=f'SkLearn model with MV_Opt Prediction {dropdown[stock]}',textposition='bottom center'))
        trace3.append(
          go.Scatter(x=df[(df["Stock"] == stock) & (df["Model"] == "DIY") & (df["Method"] == "MV_Opt")]["trunc_time"],
                     y=df[(df["Stock"] == stock) & (df["Model"] == "DIY") & (df["Method"] == "MV_Opt")]["Predictions"],
                     mode='lines', opacity=0.6,
                     name=f'DIY model with MV_Opt  Prediction {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2, trace3]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Actual and Predictions Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1 Month', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6 Month', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (VND)"})}
    return figure
 
if __name__=='__main__':
	app.run_server(debug=True)