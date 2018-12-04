# 方案一  使用分类方式模型
import Geohash
import datetime
import pandas as pd
import numpy as np
import collections

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
    code = Geohash.encode(lat, lon, 6)
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
    # 保存geohash下的中位数
    hash_lists = dict()
    start_tmp = data[['start_lat', 'start_lon']]
    start_geohash = []
    for start_loc in start_tmp.values:
        hash_str = convert_loc_to_geo(float(start_loc[0]), float(start_loc[1]), dict())
        start_geohash.append(hash_str)
        hash_list = hash_lists.get(hash_str, [])
        hash_list.append([start_loc[0], start_loc[1]])
    data['start_geohash'] = start_geohash

    end_tmp = data[['end_lat', 'end_lon']]
    end_geohash = []
    if is_train_data:
        for end_loc in end_tmp.values:
            hash_str = convert_loc_to_geo(float(end_loc[0]), float(end_loc[1]), geo_dic)
            end_geohash.append(hash_str)
            hash_list = hash_lists.get(hash_str, [])
            hash_list.append([end_loc[0], end_loc[1]])
        data['end_geohash'] = end_geohash

    medians = dict()
    for key in hash_lists.keys():
        lst = hash_lists[key]
        median_lat = np.array(lst)[:, 0]
        median_lon = np.array(lst)[:, 1]
        medians[key] = str(median_lat) + ',' + str(median_lon)
    pd.DataFrame(hash_lists).to_csv('medians.csv')

    count = collections.Counter(start_geohash + end_geohash).most_common()
    # geohash转编号
    dictionary = dict()
    for item, _ in count:
        dictionary[item] = len(dictionary)

    start_classifications = []
    for s in start_geohash:
        start_classifications.append(dictionary[s])
    data['start_classifications'] = start_classifications

    end_classifications = []
    if is_train_data:
        for e in end_geohash:
            end_classifications.append(dictionary[e])
        data['end_classification'] = end_classifications

    return data


def convert_to_lable(train, test):
    from sklearn.cluster import DBSCAN
    trL = train.shape[0] * 2
    X = np.concatenate([train[['start_lat', 'start_lon']].values,
                        train[['end_lat', 'end_lon']].values,
                        test[['start_lat', 'start_lon']].values])
    db = DBSCAN(eps=5e-4, min_samples=3, p=1, leaf_size=10, n_jobs=-1).fit(X)
    labels = db.labels_
    n_clusters_ = len(set(labels))
    print('Estimated number of clusters: %d' % n_clusters_)

    info = pd.DataFrame(X[:trL, :], columns=['lat', 'lon'])
    info['block_id'] = labels[:trL]
    clear_info = info.loc[info.block_id != -1, :]
    print('The number of miss start block in train data', (info.block_id.iloc[:trL // 2] == -1).sum())
    print('The number of miss end block in train data', (info.block_id.iloc[trL // 2:] == -1).sum())
    # 测试集聚类label
    test_info = pd.DataFrame(X[trL:, :], columns=['lat', 'lon'])
    test_info['block_id'] = labels[trL:]
    print('The number of miss start block in test data', (test_info.block_id == -1).sum())
    train['start_block'] = info.block_id.iloc[:trL // 2].values
    train['end_block'] = info.block_id.iloc[trL // 2:].values
    test['start_block'] = test_info.block_id.values
    good_train_idx = (train.start_block != -1) & (train.end_block != -1)
    print('The number of good training data', good_train_idx.sum())
    good_train = train.loc[good_train_idx, :]
    print('saving new train & test data')
    good_train.to_csv('good_train.csv', index=None)
    test.to_csv('good_test.csv', index=None)


# train = pd.read_csv('train_new.csv')
# geo_dic = dict()
# train = data_convert(train, geo_dic, True)
# train.to_csv('train_new_1.csv')
#

# test = pd.read_csv('test_new.csv')
# test = data_convert(test, None, False)
# test.to_csv('test_new_1.csv')
# with open('geo_dic.txt', 'w')as f:
#     f.writelines(str(geo_dic))


# train = pd.read_csv('train_new.csv', low_memory=False)
# test = train[train['start_time'] > '2018-07-01 00:00:00']
# train = train[train['start_time'] <= '2018-07-01 00:00:00']
# convert_to_lable(train, test)

from sklearn.naive_bayes import MultinomialNB

# df = pd.read_csv('good_train.csv', low_memory=False)
# df = data_convert(df, dict())
# df.to_csv('good_train_new.csv')
df = pd.read_csv('good_train_new.csv', low_memory=False)
clf = MultinomialNB()
X = df[['week_day', 'hour', 'is_holiday', 'start_classifications']]
y = df['end_block']
clf.fit(X, y)
print(clf.class_log_prior_)
