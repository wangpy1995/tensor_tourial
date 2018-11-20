# 方案一  使用分类方式模型
import Geohash
import datetime
import pandas as pd
import numpy as np

# 节假日
special_holiday = ['2018-01-01'] + ['2018-02-%d' % d for d in range(15, 22)] + \
                  ['2018-04-%2d' % d for d in range(5, 8)] + \
                  ['2018-04-%d' % d for d in range(29, 31)] + ['2018-05-01'] + \
                  ['2018-06-%d' % d for d in range(16, 19)] + \
                  ['2018-09-%d' % d for d in range(22, 25)] + \
                  ['2018-10-%2d' % d for d in range(1, 8)]
# 特殊工作日
special_workday = ['2018-02-%d' % d for d in [11, 24]] + \
                  ['2018-04-08'] + ['2018-04-28'] + \
                  ['2018-09-%d' % d for d in range(29, 31)]


def convert_loc_to_geo(lat, lon, geo_dic):
    # 损失部分精度，以geohash值作为类别
    code = Geohash.encode(lat, lon, 12)
    # 保存经纬度，用于计算该分类下的中位数
    if geo_dic.get(code) is None:
        geo_dic[code] = [(lat, lon)]
    else:
        geo_dic[code].append((lat, lon))
    return code


# 追加星期几，小时，是否节假日几个属性
def data_convert(data, geo_dic, is_train_data=True):
    # 节假日
    is_holiday = []
    date = pd.to_datetime(data['start_time'])
    data['week_day'] = date.dt.weekday + 1
    data['hour'] = date.dt.hour

    # 区分节假日与工作日
    for t in data['start_time']:
        ymd = t.split(' ')[0]
        day = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').day
        if ymd in special_holiday or (ymd not in special_workday and day > 5):
            is_holiday.append(True)
        else:
            is_holiday.append(False)
    data['is_holiday'] = is_holiday

    # 经纬度转geohash
    start_tmp = data[['start_lat', 'start_lon']]
    start_geohash = []
    for stat_loc in start_tmp.values:
        start_geohash.append(convert_loc_to_geo(float(stat_loc[0]), float(stat_loc[1]), dict()))
    data['start_geohash'] = start_geohash
    end_tmp = data[['end_lat', 'end_lon']]
    if is_train_data:
        end_geohash = []
        for end_loc in end_tmp.values:
            end_geohash.append(convert_loc_to_geo(float(end_loc[0]), float(end_loc[1]), geo_dic))
        data['end_geohash'] = end_geohash
    return data


df = pd.read_csv('train_new.csv')
geo_dic = dict()
df = data_convert(df, geo_dic, True)
df.to_csv('train_new_1.csv')
