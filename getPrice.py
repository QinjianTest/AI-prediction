import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# 黄金的行情数据集 'gold_prices.csv'，下载地址https://www.investing.com/commodities/gold-historical-data
data = pd.read_csv('gold_prices.csv')

# 清理和准备数据
data['Price'] = data['Price'].apply(lambda x: float(x.replace(',', '')))
data['Date'] = pd.to_datetime(data['Date'])

# 打印当天的实际价格
today = datetime.now().strftime('%Y-%m-%d')
today_price = data.loc[data['Date'] == today, 'Price']

if not today_price.empty:
    print(f"Today's Price ({today}): {today_price.values[0]:.2f}")
else:
    print(f"No data for today ({today}).")

# 将日期转换为时间戳序号
data['Date'] = data['Date'].map(pd.Timestamp.toordinal)

# 特征和目标变量
X = data[['Date']]
y = data['Price']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 进行测试集预测
predictions = model.predict(X_test)

# 计算误差
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# 打印过去5天的实际价格
print("\nPast 5 Days Prices:")
past_dates = pd.date_range(end=datetime.now() - timedelta(days=1), periods=5)
for date in past_dates:
    date_ordinal = pd.Timestamp(date).toordinal()
    if date_ordinal in X.values:
        actual_price = y[X['Date'] == date_ordinal].values[0]
        print(f"Date: {date.strftime('%Y-%m-%d')}, Actual Price: {actual_price:.2f}")

# 预测未来日期的价格
future_dates = pd.date_range(start=datetime.now() + timedelta(days=1), periods=10)  # 预测未来10天
future_dates_ordinal = pd.DataFrame(future_dates.map(pd.Timestamp.toordinal), columns=['Date'])

# 使用DataFrame进行预测
future_predictions = model.predict(future_dates_ordinal)

# 输出未来日期的预测结果
print("\nFuture Predictions:")
for date, prediction in zip(future_dates, future_predictions):
    print(f'Date: {date.strftime("%Y-%m-%d")}, Predicted Price: {prediction:.2f}')
