import pandas as pd


def load_dat(dataPath, heads, sep):
    #读取数据
    df = pd.read_table(dataPath, sep=sep, header=None, engine='python')
    df.columns = (heads)
    return df

class DataLoader():
    def __init__(self,GPS_dir,IMU_dir,GPS_date_index,IMU_date_index):
        self.GPS_data = load_dat(GPS_dir,GPS_date_index,"\\s+")
        self.IMU_data = load_dat(IMU_dir,IMU_date_index,"\\s+")

        #TODO: 数据处理 单位

    
