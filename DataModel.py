from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig()
# logging.getLogger('sqlalchemy.engine').setLevel(logging.ERROR)
logging.getLogger('sqlalchemy.engine.Engine').disabled = True

Base = declarative_base()

# column_names = ["ID", "Date", "Detail", "Ball_1", "Ball_2", "Ball_3", "Ball_4", "Ball_5", "Ball_6", "Ball_7", "Total", "Curent", "Top_Hit", "Top_Amount", "Sec_Hit", "Sec_Amount"]  # 假设的列名
class Ssq(Base):
    __tablename__ = 'ssq'
    ID = Column(String, primary_key=True)
    Date = Column(String) #0
    Week = Column(String) #1
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


def ConnectDB(name):
    engine = create_engine('sqlite:///' + name, echo=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


def LoadData(name):
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    session = ConnectDB(name)
    df = pd.read_sql('SELECT * FROM ssq order by Date ASC', session.bind)
    session.close()
    return df

def load_ssq_red():
    table = LoadData("data/ssq.db")
    reds = table[["Ball_1", "Ball_2", "Ball_3", "Ball_4", "Ball_5", "Ball_6"]].values
    return reds - 1

def load_ssq_blue_diff():
    table = LoadData("data/ssq.db")
    table['diff'] = table['Ball_7'].diff()
    return table['Ball_7'], table['diff']

def load_ssq_red_diff():
    table = LoadData("data/ssq.db")
    return table[['Ball_1', 'Ball_2', 'Ball_3', 'Ball_4', 'Ball_5', 'Ball_6']], table[['Ball_1', 'Ball_2', 'Ball_3', 'Ball_4', 'Ball_5', 'Ball_6']].diff()

def load_ssq_blue():
    table = LoadData("data/ssq.db")
    table["odd_even"] = table["Ball_7"] % 2
    table["big_small"] = (table["Ball_7"] > 8).astype(int)

    table["step"] = table['Ball_7'].diff()
    # table['step'].fillna(0, inplace=True) 
    table.fillna({'step': 0}, inplace=True)
    # table.fillna('step', 0, inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    table['step'] = scaler.fit_transform(table['step'].values.reshape(-1, 1))

    # table["Ball_7"] = table["Ball_7"] -1
    categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    one_hot_encoded = pd.get_dummies(table['Ball_7'], prefix='Ball_7')
    column = []
    for category in categories:
        column_name = f'Ball_7_{category}'
        column.append(column_name)
        if column_name not in one_hot_encoded.columns:
            one_hot_encoded[column_name] = 0
    one_hot_encoded = one_hot_encoded[sorted(one_hot_encoded.columns)]
    table = table.join(one_hot_encoded)
    column.append('odd_even')
    column.append('big_small')
    column.append('step')
    
    return table[column], table["Ball_7"] - 1

    # return blue - 1

def prepare_data_all(inputs, targets, window_size, input_size, output_size):
    window_inputs = np.zeros((len(inputs) - window_size, window_size, input_size), dtype=np.int64)
    window_targets = np.zeros((len(inputs) - window_size, output_size), dtype=np.int64)
    for i in range(len(inputs) - window_size):
        window_inputs[i] = inputs[i: i + window_size]
        window_targets[i] = targets[i + window_size]
    return window_inputs, window_targets

def prepare_data(inputs, targets, window_size, input_size, output_size, train_percentage = 0.8):
    train_len = int(len(inputs) * train_percentage)
    # test_len = len(inputs) - train_len
    
    window_inputs = np.zeros((len(inputs) - window_size, window_size, input_size), dtype=np.int64)
    window_targets = np.zeros((len(inputs) - window_size, output_size), dtype=np.int64)
    for i in range(len(inputs) - window_size):
        window_inputs[i] = inputs[i: i + window_size]
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
    print (load_ssq_blue())
