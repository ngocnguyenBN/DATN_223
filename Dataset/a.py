import pandas as pd

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

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Sử dụng Random Forest Regressor để dự đoán biến mục tiêu
from sklearn.ensemble import RandomForestRegressor

X_train = train_data.drop(['trunc_time','open_price','high_price','low_price' ,'volume' ,'close_price' ], axis=1)
# X_train = train_data.drop(['open_price'], axis=1)
# X_train = train_data.drop(['high_price'], axis=1)
# X_train = train_data.drop(['low_price'], axis=1)
# X_train = train_data.drop(['volume'], axis=1)
y_train = train_data['close_price']

print(X_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Đánh giá kết quả trên tập kiểm tra
X_test = test_data.drop(['trunc_time','open_price','high_price','low_price','volume' ,'close_price' ], axis=1)
y_test = test_data['close_price']
print("y_test")
print(test_data)
y_pred = rf.predict(X_test)
print("y_test",y_test)
print("y_pred",y_pred)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f'Mean squared error: {mse}')
