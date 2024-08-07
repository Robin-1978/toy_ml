import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta

# 从CSV文件加载数据，假设数据格式为 id,date,red_ball1,red_ball2,red_ball3,red_ball4,red_ball5,red_ball6,blue_ball
data = pd.read_csv('./data/ssq/data.csv')
data = data.iloc[::-1]

print(data)

# 计算每期的具体日期，假设从第一期开始，根据每逢星期二、四、日的规则计算日期
start_date = datetime(2003, 2, 16)  # 第一期的日期
periods_per_week = 3  # 每逢星期二、四、日
date_list = []
current_date = start_date

# 根据期号计算每期的日期
for i in range(len(data)):
    date_list.insert(0, current_date)  # 在最前面插入日期，实现逆序
    if (i + 1) % periods_per_week == 0:
        current_date += timedelta(days=3)  # 下一期的日期距离当前日期3天
    else:
        current_date += timedelta(days=2)  # 下一期的日期距离当前日期2天

# 将日期和蓝球号码放入DataFrame
blue_ball_data = pd.DataFrame({'ds': date_list, 'y': data['蓝球']})

# 创建Prophet模型
model = Prophet()
model.fit(blue_ball_data)

# 构建待预测的未来时间序列
future = model.make_future_dataframe(periods=1)  # 预测未来30天的数据

# 进行预测
forecast = model.predict(future)

# 打印预测结果
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# 绘制预测结果
# fig = model.plot(forecast)