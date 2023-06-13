import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
 
import numpy as np
 
import matplotlib.pyplot as plt
 
import time
import psutil
from multiprocessing import Process, Manager
from joblib import Parallel, delayed
import math

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 

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
   
    html.H1("Random Forest Model Comparison", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='Train Model',children=[
            html.Label("Number of Estimators:"),
            dcc.Input(id='n_estimators', type='number', value=160),
            html.Br(),
            html.Label("Number of Max Depth:"),
            dcc.Input(id='max_depth', type='number', value=26),
            html.Br(),
            html.Label("Number of Min Sample Leaf:"),
            dcc.Input(id='min_samples_leaf', type='number', value=2),
            html.Br(),
            html.Label("Number of Min Sample Split:"),
            dcc.Input(id='min_samples_split', type='number', value=10),
            html.Br(),
            html.Label("Number of Max Feature:"),
            dcc.Input(id='max_features', type='number', value=3),
            html.Br(),
            html.Label("Training Size:"),
            dcc.Input(id='training_size', type='number', value=75),
            html.Br(),
            html.Button('Run Models', id='button'),
            html.Hr(),
            html.Div(id='output-container')
        ]),

        dcc.Tab(label='Result of Trained Model', children=[
            html.Div([
                html.H1("Stocks Actual vs Predictions by only RF", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'STB', 'value': 'STB'},
                                      {'label': 'HAG','value': 'HAG'}], 
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
 

data = pd.read_csv("./Dataset/STB.csv")
data.rename(columns={"trunc_time":"Date","open_price":"open","high_price":"high","low_price":"low","close_price":"Close"}, inplace= True)

from sklearn.preprocessing import MinMaxScaler
closedf = data[['Date','Close']]
close_stock = closedf.copy()
del closedf['Date']
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
time_step = 20


# Function to run a Random Forest model
def run_sk(max_depth, min_samples_leaf, min_samples_split,n_estimators, max_features, _training_size):
    training_size=int(len(closedf) * _training_size / 100)
    test_size=len(closedf)-training_size
    train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]

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

    start_time = time.time()
    model = RandomForestRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,n_estimators=n_estimators, max_features=max_features)
    model.fit(X_train, y_train)
    execution_time = time.time() - start_time
    ram_usage = psutil.Process().memory_info().rss / 1024 ** 2  # RAM usage in MB

    # Predict on the test set
    RF_train_predict=model.predict(X_train)
    RF_test_predict=model.predict(X_test)

    RF_train_predict = RF_train_predict.reshape(-1,1)
    RF_test_predict = RF_test_predict.reshape(-1,1)

    RF_train_predict = scaler.inverse_transform(RF_train_predict)
    RF_test_predict = scaler.inverse_transform(RF_test_predict)

    RF_original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 

    RF_RMSE_test = math.sqrt(mean_squared_error(RF_original_ytest,RF_test_predict))
    RF_r2_test = r2_score(RF_original_ytest, RF_test_predict)

    look_back=time_step
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(RF_train_predict)+(look_back*2)+1:len(closedf)-1, :] = RF_test_predict
    close_stock['Predictions']=testPredictPlot

    return (close_stock, execution_time, ram_usage, RF_r2_test, RF_RMSE_test, 1)

def run_diy(max_depth, min_samples_leaf, min_samples_split,n_estimators, max_features, _training_size):
    training_size=int(len(closedf) * _training_size / 100)
    test_size=len(closedf)-training_size
    train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]

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
    class DIYRandomForestRegressor:
        def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, subsample_ratio=0.8, n_jobs=-1, random_state=42):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.max_features = max_features
            self.subsample_ratio = subsample_ratio
            self.trees = []
            self.feature_importances_ = None
            self.n_jobs = n_jobs
            self.random_state = random_state
            
        def fit(self, X, y):
            n_samples = len(X)
            n_subsample = int(self.subsample_ratio * n_samples)
            self.feature_importances_ = np.zeros(X.shape[1])
            
            def fit_tree(i):
                subsample_indices, oob_indices = self._bootstrap_sampling(n_samples, n_subsample)  # Apply bootstrap sampling
                subsample_X = X[subsample_indices]
                subsample_y = y[subsample_indices]
                
                if self.random_state is not None:
                    np.random.seed(self.random_state + i)  # Set random seed for each tree
                    
                tree = DIYDecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, 
                                            min_samples_leaf=self.min_samples_leaf, max_features=self.max_features, random_state=self.random_state)
                tree.fit(subsample_X, subsample_y)
                
                return tree, subsample_indices
            
            results = Parallel(n_jobs=self.n_jobs)(delayed(fit_tree)(i) for i in range(self.n_estimators))
            
            self.trees = [result[0] for result in results]
            self.feature_importances_ = np.sum([tree.feature_importances_ for tree, _ in results], axis=0)
            
        def predict(self, X):
            return np.mean([tree.predict(X) for tree in self.trees], axis=0)
        
        def _bootstrap_sampling(self, n_samples, n_subsample):
            subsample_indices = np.random.choice(n_samples, size=n_subsample, replace=True)  # Subsampling with Replacement
            oob_indices = np.array(list(set(range(n_samples)) - set(subsample_indices)))  # Out-of-Bag indices
            return subsample_indices, oob_indices


    class DIYDecisionTreeRegressor:
        def __init__(
            self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None
        ):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.max_features = max_features
            self.tree = None
            self.feature_importances_ = None
            self.random_state = random_state

        def fit(self, X, y):
            if self.random_state is not None:
                np.random.seed(self.random_state)
                
            self.tree = self._build_tree(X, y)
            self.feature_importances_ = self._calculate_feature_importances(X)
            
        def predict(self, X):
            return np.array([self._traverse_tree(x, self.tree) for x in X])
        
        def _build_tree(self, X, y, depth=0):
            node = Node()
            
            if depth == self.max_depth or len(X) < self.min_samples_split or len(np.unique(y)) == 1:
                node.is_leaf = True
                node.prediction = np.mean(y)
                return node
            
            split_feature, split_value = self._find_best_split(X, y)
            
            if split_feature is None:
                node.is_leaf = True
                node.prediction = np.mean(y)
                return node
            
            left_indices = X[:, split_feature] <= split_value
            right_indices = X[:, split_feature] > split_value

            node.split_feature = split_feature
            node.split_value = split_value

            node.left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
            node.right = self._build_tree(X[right_indices], y[right_indices], depth + 1)

            return node
        
        def _find_best_split(self, X, y):
            best_score = float('inf')
            best_split_feature = None
            best_split_value = None
            
            n_features = X.shape[1]
            if self.max_features is not None and self.max_features < n_features:
                feature_indices = np.random.choice(n_features, size=self.max_features, replace=False)
            else:
                feature_indices = np.arange(n_features)
            
            for feature in feature_indices:
                unique_values = np.unique(X[:, feature])
                split_values = (unique_values[:-1] + unique_values[1:]) / 2.0
                
                for value in split_values:
                    left_indices = X[:, feature] <= value
                    right_indices = X[:, feature] > value
                    score = self._calculate_split_score(y, left_indices, right_indices)
                    
                    if score < best_score:
                        best_score = score
                        best_split_feature = feature
                        best_split_value = value
            
            return best_split_feature, best_split_value
        
        def _calculate_split_score(self, y, left_indices, right_indices):
            left_y = y[left_indices]
            right_y = y[right_indices]
            
            left_score = self._calculate_variance(left_y)
            right_score = self._calculate_variance(right_y)
            
            n_left = len(left_y)
            n_right = len(right_y)
            total_samples = n_left + n_right
            
            split_score = (n_left / total_samples) * left_score + (n_right / total_samples) * right_score
            return split_score
        
        def _calculate_variance(self, y):
            if len(y) == 0:
                return 0.0
            mean = np.mean(y)
            variance = np.mean((y - mean) ** 2)
            return variance
        
        def _traverse_tree(self, x, node):
            if node.is_leaf:
                return node.prediction
            else:
                if x[node.split_feature] <= node.split_value:
                    return self._traverse_tree(x, node.left)
                else:
                    return self._traverse_tree(x, node.right)
                
        def _calculate_feature_importances(self, X):
            feature_importances = np.zeros(X.shape[1])  # Adjust the shape of feature_importances array
            # Calculate feature importances based on tree structure, node splits, or other criteria
            # Assign importance values to each feature
            return feature_importances


    class Node:
        def __init__(self):
            self.is_leaf = False
            self.prediction = None
            self.split_feature = None
            self.split_value = None
            self.left = None
            self.right = None


    start_time = time.time()
    model = DIYRandomForestRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,n_estimators=n_estimators, max_features=max_features)
    model.fit(X_train, y_train)
    execution_time = time.time() - start_time
    ram_usage = psutil.Process().memory_info().rss / 1024 ** 2  # RAM usage in MB

    # Predict on the test set
    RF_train_predict=model.predict(X_train)
    RF_test_predict=model.predict(X_test)

    RF_train_predict = RF_train_predict.reshape(-1,1)
    RF_test_predict = RF_test_predict.reshape(-1,1)

    RF_train_predict = scaler.inverse_transform(RF_train_predict)
    RF_test_predict = scaler.inverse_transform(RF_test_predict)

    RF_original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 

    RF_RMSE_test = math.sqrt(mean_squared_error(RF_original_ytest,RF_test_predict))
    RF_r2_test = r2_score(RF_original_ytest, RF_test_predict)

    look_back=time_step
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(RF_train_predict)+(look_back*2)+1:len(closedf)-1, :] = RF_test_predict
    close_stock['Predictions_DIY']=testPredictPlot
    return (close_stock, execution_time, ram_usage, RF_r2_test, RF_RMSE_test, 2)

# Callback to run the models and display the results
@app.callback(Output('output-container', 'children'),
              [Input('button', 'n_clicks')],
              [dash.dependencies.State('max_depth', 'value'),
               dash.dependencies.State('min_samples_leaf', 'value'),
               dash.dependencies.State('min_samples_split', 'value'),
               dash.dependencies.State('n_estimators', 'value'),
               dash.dependencies.State('max_features', 'value'),
               dash.dependencies.State('training_size', 'value')])
def run_models(n_clicks, max_depth, min_samples_leaf, min_samples_split,n_estimators, max_features, training_size):
    if n_clicks:
        print(training_size)
        model_diy, exec_time_diy, ram_usage_diy, accuracy_diy, rmse_diy, _ = run_diy(max_depth, min_samples_leaf, min_samples_split,n_estimators, max_features, training_size)
        model_sk, exec_time_sk, ram_usage_sk, accuracy_sk, rmse_sk, _ = run_sk(max_depth, min_samples_leaf, min_samples_split,n_estimators, max_features, training_size)

        trace1_sk = []
        trace2_sk = []
        trace1_sk.append(
            go.Scatter(x=model_sk['Date'],
                        y=model_sk['Close'],
                        mode='lines', opacity=0.7, 
                        name=f'Actual STB',textposition='bottom center'))
        trace2_sk.append(
            go.Scatter(x=model_sk['Date'],
                        y=model_sk['Predictions'],
                        mode='lines', opacity=0.6,
                        name=f'Prediction STB',textposition='bottom center'))
        traces_sk = [trace1_sk, trace2_sk]
        traces_sk = [val for sublist in traces_sk for val in sublist]
        print("model_sk", model_diy)
        print("model_diy", model_diy)
        trace1_diy = []
        trace2_diy = []
        trace1_diy.append(
            go.Scatter(x=model_diy['Date'],
                        y=model_diy['Close'],
                        mode='lines', opacity=0.7, 
                        name=f'Actual STB',textposition='bottom center'))
        trace2_diy.append(
            go.Scatter(x=model_diy['Date'],
                        y=model_diy['Predictions_DIY'],
                        mode='lines', opacity=0.6,
                        name=f'Prediction STB',textposition='bottom center'))
        traces_diy = [trace1_diy, trace2_diy]
        traces_diy = [val for sublist in traces_diy for val in sublist]

        trace1_total = []
        trace2_total = []
        trace3_total = []
        trace1_total.append(
            go.Scatter(x=model_sk['Date'],
                        y=model_sk['Close'],
                        mode='lines', opacity=0.7, 
                        name=f'Actual STB',textposition='bottom center'))
        trace2_total.append(
            go.Scatter(x=model_sk['Date'],
                        y=model_sk['Predictions'],
                        mode='lines', opacity=0.6,
                        name=f'SK RF Prediction STB',textposition='bottom center'))         
        trace3_total.append(
                    go.Scatter(x=model_diy['Date'],
                                y=model_diy['Predictions_DIY'],
                                mode='lines', opacity=0.6,
                                name=f'DIY RF Prediction STB',textposition='bottom center'))
        traces_total = [trace1_total, trace2_total, trace3_total]
        traces_total = [val for sublist in traces_total for val in sublist]
        model_sk_output = html.Div([
            dcc.Graph(id='scatter-plot', figure={'data': traces_sk,
                'layout': go.Layout(colorway=["#2dcde8", '#00cc96', '#ef553b', 
                                            '#FF7400', '#FFF400', '#FF0056'],
                height=600,
                title=f"Scikit Learn Random Forest",
                xaxis={"title":"Date",
                    'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                        'step': 'month', 
                                                        'stepmode': 'backward'},
                                                        {'count': 12, 'label': '1Y', 
                                                        'step': 'month', 
                                                        'stepmode': 'backward'},
                                                        {'step': 'all'}])},
                    'rangeslider': {'visible': True}, 'type': 'date'},
                yaxis={"title":"Price (USD)"})}
            ),
            html.P(f"Execution Time: {exec_time_sk:.2f} seconds"),
            html.P(f"RAM Usage: {ram_usage_sk:.2f} MB"),
            html.P(f"R2 Score: {accuracy_sk:.2f}"),
            html.P(f"RSME: {rmse_sk:.2f}"),
            html.Hr()
        ])

        model_diy_output = html.Div([
            dcc.Graph(id='scatter-plot', figure={'data': traces_diy,
                'layout': go.Layout(colorway=["#2dcde8", '#00cc96', '#ef553b', 
                                            '#FF7400', '#FFF400', '#FF0056'],
                height=600,
                title=f"DIY Random Forest",
                xaxis={"title":"Date",
                    'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                        'step': 'month', 
                                                        'stepmode': 'backward'},
                                                        {'count': 12, 'label': '1Y', 
                                                        'step': 'month', 
                                                        'stepmode': 'backward'},
                                                        {'step': 'all'}])},
                    'rangeslider': {'visible': True}, 'type': 'date'},
                yaxis={"title":"Price (USD)"})}
            ),
            html.P(f"Execution Time: {exec_time_diy:.2f} seconds"),
            html.P(f"RAM Usage: {ram_usage_diy:.2f} MB"),
            html.P(f"R2 Score: {accuracy_diy:.2f}"),
            html.P(f"RSME: {rmse_diy:.2f}"),
            html.Hr()
        ])

        model_total_output = html.Div([
            dcc.Graph(id='scatter-plot', figure={'data': traces_total,
                'layout': go.Layout(colorway=["#2dcde8", '#00cc96', '#ef553b', 
                                            '#FF7400', '#FFF400', '#FF0056'],
                height=600,
                title=f"Comparison of 2 models",
                xaxis={"title":"Date",
                    'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                        'step': 'month', 
                                                        'stepmode': 'backward'},
                                                        {'count': 12, 'label': '1Y', 
                                                        'step': 'month', 
                                                        'stepmode': 'backward'},
                                                        {'step': 'all'}])},
                    'rangeslider': {'visible': True}, 'type': 'date'},
                yaxis={"title":"Price (USD)"})}
            )
        ])

        return html.Div([model_sk_output, model_diy_output, model_total_output])
 
 
 
@app.callback(Output('none', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"STB": "STB","HAG": "HAG"}
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
              'layout': go.Layout(colorway=["#2dcde8", '#00cc96', '#ef553b', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Actual and Predictions Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1 Month', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                        {'count': 12, 'label': '1Y', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (VND)"})}
    return figure

@app.callback(Output('mv', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"STB": "STB","HAG": "HAG"}
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
              'layout': go.Layout(colorway=["#2dcde8", '#00cc96', '#ef553b', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Actual and Predictions Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1 Month', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                        {'count': 12, 'label': '1Y', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (VND)"})}
    return figure

@app.callback(Output('opt', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"STB": "STB","HAG": "HAG"}
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
              'layout': go.Layout(colorway=["#2dcde8", '#00cc96', '#ef553b', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Actual and Predictions Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1 Month', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                        {'count': 12, 'label': '1Y', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (VND)"})}
    return figure

@app.callback(Output('mv-opt', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"STB": "STB","HAG": "HAG"}
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
              'layout': go.Layout(colorway=["#2dcde8", '#00cc96', '#ef553b', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Actual and Predictions Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1 Month', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                        {'count': 12, 'label': '1Y', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (VND)"})}
    return figure
 
if __name__=='__main__':
	app.run_server(debug=True)