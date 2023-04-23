import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
import math
segment_size = 100

# Calculate the number of segments in the dataset
num_segments = 17
# Đọc vào tập dữ liệu bị phân mảnh ngang
data_chunks = []
for i in range(num_segments):
    chunk = pd.read_csv(f'STB.csv')
    data_chunks.append(chunk)
data = pd.concat(data_chunks, ignore_index=True)

# Áp dụng moving average với cửa sổ 5 và lưu vào cột 'ma_5'
data['ma_5'] = data['close_price'].rolling(window=5).mean()

# Xóa bỏ các dòng có giá trị null do moving average
data.dropna(inplace=True)
print(type(data))

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)
print(train_data.shape, test_data.shape)
print(train_data, test_data)

# Sử dụng Random Forest Regressor để dự đoán biến mục tiêu
from sklearn.ensemble import RandomForestRegressor
print("train_data", train_data)
X_train = train_data.drop(['trunc_time','open_price','high_price','low_price' ,'volume' ,'close_price' ], axis=1)
# X_train = train_data.drop(['open_price'], axis=1)
# X_train = train_data.drop(['high_price'], axis=1)
# X_train = train_data.drop(['low_price'], axis=1)
# X_train = train_data.drop(['volume'], axis=1)
y_train = train_data['close_price']

print("X_train, y_train")
print(X_train, y_train.shape)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Đánh giá kết quả trên tập kiểm tra
X_test = test_data.drop(['trunc_time','open_price','high_price','low_price','volume' ,'close_price' ], axis=1)
y_test = test_data['close_price']
print("y_test")
print(X_test, y_test)
y_pred_train=rf.predict(X_train)
y_pred = rf.predict(X_test)
print("y_test",y_test)
print("y_pred",y_pred)

from sklearn.metrics import mean_squared_error

RF_RMSE_train = math.sqrt(mean_squared_error(y_train,y_pred_train))
RF_MSE_train = mean_squared_error(y_train,y_pred_train)
RF_MAE_train = mean_absolute_error(y_train,y_pred_train)

RF_RMSE_test = math.sqrt(mean_squared_error(y_test, y_pred))
RF_MSE_test = mean_squared_error(y_test, y_pred)
RF_MAE_test = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean squared error: {mse}')
print("Train data RMSE: ", RF_RMSE_train)
print("Train data MSE: ", RF_MSE_train)
print("Test data MAE: ", RF_MAE_train)
print("Test data RMSE: ", RF_RMSE_test)
print("Test data MSE: ", RF_MSE_test)
print("Test data MAE: ", RF_MAE_test)
