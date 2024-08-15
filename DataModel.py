from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import logging
import numpy as np

logging.basicConfig()
# logging.getLogger('sqlalchemy.engine').setLevel(logging.ERROR)
logging.getLogger('sqlalchemy.engine.Engine').disabled = True

Base = declarative_base()

#column_names = ["ID", "Date", "Detail", "Ball_1", "Ball_2", "Ball_3", "Ball_4", "Ball_5", "Ball_6", "Ball_7", "Total", "Curent", "Top_Hit", "Top_Amount", "Sec_Hit", "Sec_Amount"]  # 假设的列名
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
    red_train, red_target, blue_train, blue_target = PrepareSSQ(3)
    print(red_train.shape, red_target.shape, blue_train.shape, blue_target.shape)