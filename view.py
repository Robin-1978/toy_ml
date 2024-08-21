import DataModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objs as go

def view_data():
    df = DataModel.LoadData("data/ssq.db")
    # dates = df["Date"]
    # data = pd.Series(df["Ball_7"], index=dates)
    # data = pd.Series(data=np.sin(np.linspace(0, 10, len(dates))) + np.random.normal(scale=0.5, size=len(dates)), index=dates)
    df.set_index("Date", inplace=True)
    # 创建 Plotly 图表
    fig = go.Figure()

    # 添加每个数据列的线图到同一个图中
    for i in range(1, 8):
        fig.add_trace(
            go.Scatter(x=df.index, y=df[f'Ball_{i}'], mode='lines', name=f'Ball_{i}')
        )
    
    # 更新布局
    fig.update_layout(
        title='Interactive Time Series Plot with 6 Data Points per Time',
        xaxis_title='Date',
        yaxis_title='Value',
        xaxis_rangeslider_visible=True  # 显示滚动条
    )
    
    # 显示图表
    fig.show()

if __name__ == '__main__':
    view_data()
