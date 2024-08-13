import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import logging

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
    # 创建数据库引擎
    # logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
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