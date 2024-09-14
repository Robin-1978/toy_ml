from sqlalchemy import create_engine, Column, Integer, String, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig()
# logging.getLogger('sqlalchemy.engine').setLevel(logging.ERROR)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

Base = declarative_base()


# column_names = ["ID", "Date", "Detail", "Ball_1", "Ball_2", "Ball_3", "Ball_4", "Ball_5", "Ball_6", "Ball_7", "Total", "Curent", "Top_Hit", "Top_Amount", "Sec_Hit", "Sec_Amount"]  # 假设的列名
class Ssq(Base):
    __tablename__ = "ssq"
    ID = Column(String, primary_key=True)
    Date = Column(String)  # 0
    Week = Column(String)  # 1
    Detail = Column(String)
    Ball_1 = Column(Integer)
    Ball_2 = Column(Integer)
    Ball_3 = Column(Integer)
    Ball_4 = Column(Integer)
    Ball_5 = Column(Integer)
    Ball_6 = Column(Integer)
    Ball_7 = Column(Integer)
    Total = Column(String)
    Curent = Column(String)
    TopHit = Column(String)
    TopAmount = Column(String)
    SecHit = Column(String)
    SecAmount = Column(String)


class Fc3d(Base):
    __tablename__ = "fc3d"
    ID = Column(String, primary_key=True)
    Date = Column(String)  # 日期
    Week = Column(String)  # 星期
    Detail = Column(String)  # 详细信息
    Ball_1 = Column(Integer)  # 第1个球号
    Ball_2 = Column(Integer)  # 第2个球号
    Ball_3 = Column(Integer)  # 第3个球号
    Total = Column(String)  # 销售额
    Curent = Column(String)  # 当前奖池
    TopHit = Column(String)  # 一等奖中奖注数
    TopAmount = Column(String)  # 一等奖单注奖金
    SecHit = Column(String)  # 二等奖中奖注数
    SecAmount = Column(String)  # 二等奖单注奖金


class PredictTable(Base):
    __tablename__ = "predict"
    ID = Column(Integer, primary_key=True, autoincrement=True)
    Basic = Column(Integer)
    Step = Column(Numeric)
    Predict = Column(Numeric)
    Goal = Column(Integer)
    Diff = Column(Numeric)
    Trend = Column(Integer)


def ConnectDB(name):
    engine = create_engine("sqlite:///" + name, echo=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


def LoadData(name, table_name="ssq"):
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    session = ConnectDB(name)
    df = pd.read_sql(f"SELECT * FROM {table_name} order by Date ASC", session.bind)
    session.close()
    return df


def scale_columns(df, columns):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[columns] = scaler.fit_transform(df[columns])
    return df


def load_ssq():
    table = LoadData("data/ssq.db")
    balls = table[["Ball_1", "Ball_2", "Ball_3", "Ball_4", "Ball_5", "Ball_6", "Ball_7"]]
    return balls


def scale_column(column, scaler_dict):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(column.values.reshape(-1, 1))
    scaler_dict[column.name] = scaler
    return pd.Series(scaled.flatten(), name=f"{column.name}_scale")

def make_mean(column, window_size=7):
    column = column.rolling(window_size).mean()
    return column.rename(f"{column.name}_mean")

def make_std(column, window_size=7):
    column = column.rolling(window_size).std()
    return column.rename(f"{column.name}_std")

def make_diff(column):
    column = column.diff()
    return column.rename(f"{column.name}_diff")

def make_rsi(column, window_size=7):
    diff = column.diff()
    up = diff.clip(lower=0)
    down = -1 * diff.clip(upper=0)
    ema_up = up.ewm(span=window_size).mean()
    ema_down = down.ewm(span=window_size).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.rename(f"{column.name}_rsi")
    return rsi

def make_zscore(column, window_size=7):
    column = column.rolling(window_size).mean()
    column = column.rename(f"{column.name}_zscore")
    return column

def make_bollinger_bands(column, window_size=7):
    column_mean = column.rolling(window_size).mean()
    column_std = column.rolling(window_size).std()

    upper_band = column_mean + 2 * column_std
    lower_band = column_mean - 2 * column_std
    upper_band = upper_band.rename(f"{column.name}_upper")
    lower_band = lower_band.rename(f"{column.name}_lower")
    return upper_band, lower_band

def make_macd(column, window_size1=12, window_size2=26):
    column1 = column.ewm(span=window_size1).mean()
    column2 = column.ewm(span=window_size2).mean()
    column = column1 - column2
    column = column.rename(f"{column.name}_macd")
    return column

def load_gold_features(window_size=7):
    df = pd.read_csv("data/gold_price_data.csv")

    column_names = ["High", "Low", "Open", "Close", "Volume"]
    # Dictionary to store scalers
    scaler_dict = {}
    # List to store new columns for later concatenation
    new_columns = []
    for(column_name) in column_names:
        #scaled data
        new_columns.append(scale_column(df[column_name], scaler_dict))

        # diff data
        diff_column = make_diff(df[column_name])
        new_columns.append(diff_column)
        new_columns.append(scale_column(diff_column, scaler_dict))

        ma_column = make_mean(df[column_name], window_size)
        new_columns.append(ma_column)
        new_columns.append(scale_column(ma_column, scaler_dict))


        std_column = make_std(df[column_name], window_size)
        new_columns.append(std_column)
        new_columns.append(scale_column(std_column, scaler_dict))

        rsi_column = diff_column
        rsi_column = rsi_column.rename(f"{column_name}_rsi")
        gain = (rsi_column.where(rsi_column > 0, 0)).rolling(window=window_size).mean()
        loss = (-rsi_column.where(rsi_column < 0, 0)).rolling(window=window_size).mean()
        rs = gain / loss
        rsi_column = 100 - (100 / (1 + rs))
        new_columns.append(rsi_column)
        new_columns.append(scale_column(rsi_column, scaler_dict))

        zscore_column = (df[column_name] - ma_column) / std_column
        zscore_column = zscore_column.rename(f"{column_name}_zscore")
        new_columns.append(zscore_column)
        new_columns.append(scale_column(zscore_column, scaler_dict))

    df = pd.concat(new_columns, axis=1)
    df.dropna(inplace=True)
    return df, scaler_dict

def load_ssq_features(lag=5, freq=10):
    df = LoadData("data/ssq.db")

    # Dictionary to store scalers
    scaler_dict = {}

    # List to store new columns for later concatenation
    new_columns = []

    for idx in range(1, 8):
        ball_name = f"Ball_{idx}"
        new_columns.append(scale_column(df[ball_name], scaler_dict))

        # Generate lag features
        for i in range(1, lag + 1):
            lag_column = df[ball_name].shift(i)
            lag_column = lag_column.rename(f"{ball_name}_lag_{i}")
            new_columns.append(lag_column)
            new_columns.append(scale_column(lag_column, scaler_dict))

        # Difference feature and scaling
        diff_column = df[ball_name].diff()
        diff_column = diff_column.rename(f"{ball_name}_diff")
        new_columns.append(diff_column)
        new_columns.append(scale_column(diff_column, scaler_dict))

        # Frequency feature and scaling
        freq_column = (
            df[ball_name].rolling(window=freq).apply(lambda x: pd.Series(x).value_counts().max())
        )
        freq_column = freq_column.rename(f"{ball_name}_freq")
        new_columns.append(freq_column)
        new_columns.append(scale_column(freq_column, scaler_dict))

        # Rolling mean and standard deviation
        mean_column = df[ball_name].rolling(window=freq).mean()
        mean_column = mean_column.rename(f"{ball_name}_mean")
        new_columns.append(mean_column)
        new_columns.append(scale_column(mean_column, scaler_dict))

        std_column = df[ball_name].rolling(window=freq).std()
        std_column = std_column.rename(f"{ball_name}_std")
        new_columns.append(std_column)
        new_columns.append(scale_column(std_column, scaler_dict))

        # Size, odd/even, cumulative sum, and cumulative product
        if idx < 7:
            size_column = df[ball_name].apply(lambda x: 1 if x > 16 else 0)
        else:
            size_column = df[ball_name].apply(lambda x: 1 if x > 8 else 0)
        new_columns.append(size_column.rename(f"{ball_name}_size"))

        new_columns.append(df[ball_name].mod(2).rename(f"{ball_name}_odd_even"))

        cumsum_column = df[ball_name].cumsum()
        cumsum_column = cumsum_column.rename(f"{ball_name}_cumsum")
        new_columns.append(cumsum_column)
        new_columns.append(scale_column(cumsum_column, scaler_dict))

        cumprod_column = df[ball_name].cumprod()
        cumprod_column = cumprod_column.rename(f"{ball_name}_cumprod")
        new_columns.append(cumprod_column)
        new_columns.append(scale_column(cumprod_column, scaler_dict))

    # Date-related features
    df["date"] = pd.to_datetime(df["Date"])
    month_column = df["date"].dt.month
    month_column = month_column.rename("Month")
    new_columns.append(month_column)
    new_columns.append(scale_column(month_column, scaler_dict))

    weekday_column = df["date"].dt.weekday
    weekday_column = weekday_column.rename("Weekday")
    new_columns.append(weekday_column)
    new_columns.append(scale_column(weekday_column, scaler_dict))

    day_column = df["date"].dt.day
    day_column = day_column.rename("Day")
    new_columns.append(day_column.rename("Day"))
    new_columns.append(scale_column(day_column, scaler_dict))

    # Concatenate all new columns to the original DataFrame
    df = pd.concat([df] + new_columns, axis=1)

    df.dropna(inplace=True)

    return df, scaler_dict


def load_ssq_red():
    table = LoadData("data/ssq.db")
    reds = table[["Ball_1", "Ball_2", "Ball_3", "Ball_4", "Ball_5", "Ball_6"]].values
    return reds - 1


def load_ssq_blue_diff():
    table = LoadData("data/ssq.db")
    table["diff"] = table["Ball_7"].diff()
    return table["Ball_7"], table["diff"]


def load_ssq_single_diff(num):
    table = LoadData("data/ssq.db")
    table["diff"] = table[f"Ball_{num}"].diff()
    return table[f"Ball_{num}"], table["diff"]


def load_fc3d_single_diff(num):
    table = LoadData("data/fc3d.db", "fc3d")
    table["diff"] = table[f"Ball_{num}"].diff()
    return table[f"Ball_{num}"], table["diff"]


def load_3d_features():
    table = LoadData("data/fc3d.db", "fc3d")
    new_columns = []

    for idx in range(1, 4):
        ball_name = f"Ball_{idx}"
        new_columns.append(scale_column(table[ball_name]))

        diff_column = table[ball_name].diff()
        diff_column = diff_column.rename(f"{ball_name}_diff")
        new_columns.append(diff_column)
        new_columns.append(scale_column(diff_column))
        trend_direction = diff_column.apply(lambda x: 2 if x > 0 else (0 if x < 0 else 1))
        trend_direction = trend_direction.rename(f"{ball_name}_trend")
        new_columns.append(trend_direction)

        size_column = table[ball_name].apply(lambda x: 1 if x > 4 else 0)
        new_columns.append(size_column.rename(f"{ball_name}_size"))
        new_columns.append(table[ball_name].mod(2).rename(f"{ball_name}_odd_even"))

        # Rolling mean and standard deviation
        mean_column = table[ball_name].rolling(window=10).mean()
        mean_column = mean_column.rename(f"{ball_name}_mean")
        new_columns.append(mean_column)
        new_columns.append(scale_column(mean_column))

        std_column = table[ball_name].rolling(window=10).std()
        std_column = std_column.rename(f"{ball_name}_std")
        new_columns.append(std_column)
        new_columns.append(scale_column(std_column))

    df = pd.concat([table] + new_columns, axis=1)

    df.dropna(inplace=True)

    return df


def load_ssq_red_diff():
    table = LoadData("data/ssq.db")
    return (
        table[["Ball_1", "Ball_2", "Ball_3", "Ball_4", "Ball_5", "Ball_6"]],
        table[["Ball_1", "Ball_2", "Ball_3", "Ball_4", "Ball_5", "Ball_6"]].diff(),
    )


def load_ssq_blue():
    table = LoadData("data/ssq.db")
    table["odd_even"] = table["Ball_7"] % 2
    table["big_small"] = (table["Ball_7"] > 8).astype(int)

    table["step"] = table["Ball_7"].diff()
    # table['step'].fillna(0, inplace=True)
    table.fillna({"step": 0}, inplace=True)
    # table.fillna('step', 0, inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    table["step"] = scaler.fit_transform(table["step"].values.reshape(-1, 1))

    # table["Ball_7"] = table["Ball_7"] -1
    categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    one_hot_encoded = pd.get_dummies(table["Ball_7"], prefix="Ball_7")
    column = []
    for category in categories:
        column_name = f"Ball_7_{category}"
        column.append(column_name)
        if column_name not in one_hot_encoded.columns:
            one_hot_encoded[column_name] = 0
    one_hot_encoded = one_hot_encoded[sorted(one_hot_encoded.columns)]
    table = table.join(one_hot_encoded)
    column.append("odd_even")
    column.append("big_small")
    column.append("step")

    return table[column], table["Ball_7"] - 1

    # return blue - 1


def prepare_data_all(inputs, targets, window_size, input_size, output_size):
    window_inputs = np.zeros((len(inputs) - window_size, window_size, input_size), dtype=np.int64)
    window_targets = np.zeros((len(inputs) - window_size, output_size), dtype=np.int64)
    for i in range(len(inputs) - window_size):
        window_inputs[i] = inputs[i : i + window_size]
        window_targets[i] = targets[i + window_size]
    return window_inputs, window_targets


def prepare_data(inputs, targets, window_size, input_size, output_size, train_percentage=0.8):
    train_len = int(len(inputs) * train_percentage)
    # test_len = len(inputs) - train_len

    window_inputs = np.zeros((len(inputs) - window_size, window_size, input_size), dtype=np.int64)
    window_targets = np.zeros((len(inputs) - window_size, output_size), dtype=np.int64)
    for i in range(len(inputs) - window_size):
        window_inputs[i] = inputs[i : i + window_size]
        window_targets[i] = targets[i + window_size]

    train_inputs = window_inputs[:train_len]
    test_inputs = window_inputs[train_len:]
    train_targets = window_targets[:train_len]
    test_targets = window_targets[train_len:]

    return train_inputs, train_targets, test_inputs, test_targets


def LoadSSQ():
    table = LoadData("data/ssq.db")
    blue = np.expand_dims(table["Ball_7"].values, axis=1)
    reds = table[["Ball_1", "Ball_2", "Ball_3", "Ball_4", "Ball_5", "Ball_6"]].values
    return reds, blue


def PrepareSSQ(window_size, reds, blue):
    # reds, blue = LoadSSQ()
    data_len = len(reds)
    if data_len != len(blue):
        raise ValueError("reds and blue must have the same length")
    num_samples = data_len - window_size

    red_inputs_array = np.zeros((num_samples, window_size, 6), dtype=np.int64)
    red_targets_array = np.zeros((num_samples, 6), dtype=np.int64)
    blue_inputs_array = np.zeros((num_samples, window_size, 1), dtype=np.int64)
    blue_targets_array = np.zeros((num_samples, 1), dtype=np.int64)
    for i in range(num_samples):
        start = i
        end = i + window_size
        red_inputs_array[i] = reds[start:end]
        red_targets_array[i] = reds[end]
        blue_inputs_array[i] = blue[start:end]
        blue_targets_array[i] = blue[end]

    return red_inputs_array, red_targets_array, blue_inputs_array, blue_targets_array


if __name__ == "__main__":
    # red_train, red_target, blue_train, blue_target = PrepareSSQ(3)
    # print(red_train.shape, red_target.shape, blue_train.shape, blue_target.shape)
    print(load_gold_features(3))
