import yfinance as yf
import matplotlib.pyplot as plt

# 下载 GLD 和 DXY 的历史数据
data = yf.download(['GLD'], start='2020-01-01', end='2023-01-01')

# 绘制黄金价格和美元指数的收盘价
data['Close'].plot(figsize=(10, 5))
plt.title('Gold (GLD) vs US Dollar Index (DXY)')
plt.ylabel('Price')
plt.show()