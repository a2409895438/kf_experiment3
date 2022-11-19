from DataLoader import DataLoader












if __name__ == "__main__":
    # 数据读取
    GPS_dir = "./数据/GPS.dat"
    IMU_dir = "./数据/IMU.dat"
    GPS_date_index = ["id", "UTC", "L", "lambda", "H", "V_East", "V_North", "V_Up"]
    IMU_date_index = ["id", "UTC", "wx", "wy", "wz", "fx", "fy", "fz"]
    dataLoader = DataLoader(GPS_dir,IMU_dir,GPS_date_index,IMU_date_index)
    print(dataLoader.GPS_data)
    print(dataLoader.IMU_data)