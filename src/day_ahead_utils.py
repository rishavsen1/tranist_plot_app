from pyspark.sql import functions as F
from sklearn.metrics import mean_squared_error
from src.config import *
from src import data_utils
from sklearn.preprocessing import OneHotEncoder
import joblib
import pandas as pd
import numpy as np
import random

def convert_pandas_dow_to_pyspark(pandas_dow):
    return (pandas_dow + 1) % 7 + 1

def get_past_data(spark, predict_date):
    DAY_OF_WEEK = convert_pandas_dow_to_pyspark(pd.Timestamp(predict_date).day_of_week)
    apcdata = spark.read.load(f"data/{MTA_PARQUET}")
    apcdata.createOrReplaceTempView("apc")

    # filter subset
    # TODO: Fix hard coding
    start = '2022-03-06'
    end   =' 2022-04-06'
    query = f"""
            SELECT *
            FROM apc
            WHERE (transit_date >= '{start}') AND (transit_date <= '{end}') AND (dayofweek == '{DAY_OF_WEEK}')
            """
    apcdata=spark.sql(query)
    apcdata = data_utils.remove_nulls_from_apc(apcdata)
    apcdata.createOrReplaceTempView('apcdata')
    apcdata_per_trip = data_utils.get_apc_per_trip_sparkview(spark)
    df = apcdata_per_trip.toPandas()
    df = df.sort_values(by='transit_date')
    return df
    
def generate_new_features(tdf, time_window=30, past_trips=20, target='y_reg'):
    tdf['day'] = tdf.transit_date.dt.day
    tdf['time_window'] = tdf.apply(lambda x: data_utils.get_time_window(x, time_window, row_name='arrival_time'), axis=1)
    sort2 = ['block_abbr', 'transit_date', 'arrival_time', 'route_id_direction']
    tdf = tdf.sort_values(sort2)
    tdf = tdf.dropna()
    return tdf
    
def prepare_data_for_training(df, OHE_COLUMNS, ORD_COLUMNS, add_embedding_id=True, target='y_reg', class_bins=CLASS_BINS):
    df = df[df[target] < TARGET_MAX]
    df, percentiles = data_utils.add_target_column_classification(df, target, TARGET_COLUMN_CLASSIFICATION, class_bins)
    
    ix_map = {}
    if add_embedding_id:
        for col in ORD_COLUMNS:
            ix_map[col] = data_utils.create_ix_map(df, df, col)
            df[f"{col}_ix"] = df[col].apply(lambda x: ix_map[col][x])
    df = df.drop(columns=ORD_COLUMNS)
    
    # OHE for route_id_direction
    ohe_encoder = OneHotEncoder()
    ohe_encoder = ohe_encoder.fit(df[OHE_COLUMNS])
    df[ohe_encoder.get_feature_names_out()] = ohe_encoder.transform(df[OHE_COLUMNS]).toarray()

    df = df.drop(columns=OHE_COLUMNS)
    
    return df, ix_map, ohe_encoder, percentiles

# This is hardcoded
def adjust_bins(rf_df, TARGET='y_reg', percentiles=None):
    # Train 2 separate models for bins 0, 1, 2 and 2, 3, 4
    # Adjusting y_class to incorporate Dan's request
    # Use Transit's 3 bins as a base. For the highest capacity bin, carve out everything from 55 to 75 as a 4th bin, and 75+ as a 5th bin.

    percentiles[2] = (16.0, 55.0)
    percentiles.append((56.0, 75.0))
    percentiles.append((76.0, 100.0))

    highload_df = rf_df[(rf_df[TARGET] >= percentiles[3][0]) & (rf_df[TARGET] <= percentiles[3][1])]
    rf_df.loc[highload_df.index, 'y_class'] = 3

    highload_df = rf_df[(rf_df[TARGET] >= percentiles[4][0]) & (rf_df[TARGET] <= percentiles[4][1])]
    rf_df.loc[highload_df.index, 'y_class'] = 4
    return rf_df, percentiles

# Convert to the same format as the model input
def prepare_day_ahead_for_prediction(input_df):
    ord_features = ['year', 'month', 'hour', 'day']
    cat_features = ['route_id_direction', 'is_holiday', 'dayofweek']
    train_columns = ['temperature', 'humidity', 'precipitation_intensity',
       'scheduled_headway', 'time_window', 
       'route_id_direction_14_FROM DOWNTOWN',
       'route_id_direction_14_TO DOWNTOWN', 'route_id_direction_17_FROM DOWNTOWN', 'route_id_direction_17_TO DOWNTOWN', 'route_id_direction_18_FROM DOWNTOWN', 'route_id_direction_18_TO DOWNTOWN', 'route_id_direction_19_FROM DOWNTOWN',
       'route_id_direction_19_TO DOWNTOWN', 'route_id_direction_21_NORTHBOUND','route_id_direction_21_SOUTHBOUND','route_id_direction_22_FROM DOWNTOWN','route_id_direction_22_TO DOWNTOWN','route_id_direction_23_FROM DOWNTOWN','route_id_direction_23_TO DOWNTOWN','route_id_direction_24_FROM DOWNTOWN',
       'route_id_direction_24_TO DOWNTOWN', 'route_id_direction_25_NORTHBOUND', 'route_id_direction_25_SOUTHBOUND', 'route_id_direction_28_FROM DOWNTOWN', 'route_id_direction_28_TO DOWNTOWN', 'route_id_direction_29_FROM DOWNTOWN', 'route_id_direction_29_TO DOWNTOWN', 
       'route_id_direction_34_FROM DOWNTOWN', 'route_id_direction_34_TO DOWNTOWN', 'route_id_direction_35_FROM DOWNTOWN', 'route_id_direction_35_TO DOWNTOWN', 'route_id_direction_38_FROM DOWNTOWN', 'route_id_direction_38_TO DOWNTOWN', 'route_id_direction_3_FROM DOWNTOWN',
       'route_id_direction_3_TO DOWNTOWN','route_id_direction_41_FROM DOWNTOWN','route_id_direction_41_TO DOWNTOWN','route_id_direction_42_FROM DOWNTOWN','route_id_direction_42_TO DOWNTOWN','route_id_direction_43_FROM DOWNTOWN','route_id_direction_43_TO DOWNTOWN','route_id_direction_4_FROM DOWNTOWN','route_id_direction_4_TO DOWNTOWN',
       'route_id_direction_50_FROM DOWNTOWN','route_id_direction_50_TO DOWNTOWN','route_id_direction_52_FROM DOWNTOWN','route_id_direction_52_TO DOWNTOWN','route_id_direction_55_FROM DOWNTOWN','route_id_direction_55_TO DOWNTOWN','route_id_direction_56_FROM DOWNTOWN','route_id_direction_56_TO DOWNTOWN','route_id_direction_5_FROM DOWNTOWN',
       'route_id_direction_5_TO DOWNTOWN','route_id_direction_64_FROM RIVERFRONT','route_id_direction_64_TO RIVERFRONT','route_id_direction_6_FROM DOWNTOWN',
       'route_id_direction_6_TO DOWNTOWN', 'route_id_direction_72_EDMONDSON',
       'route_id_direction_72_GRASSMERE', 'route_id_direction_75_NORTHBOUND',
       'route_id_direction_75_SOUTHBOUND', 'route_id_direction_76_LOOP',
       'route_id_direction_79_EASTBOUND', 'route_id_direction_79_NORTHBOUND',
       'route_id_direction_7_FROM DOWNTOWN',
       'route_id_direction_7_TO DOWNTOWN',
       'route_id_direction_84_FROM NASHVILLE',
       'route_id_direction_84_TO NASHVILLE',
       'route_id_direction_86_FROM NASHVILLE',
       'route_id_direction_86_TO NASHVILLE',
       'route_id_direction_8_FROM DOWNTOWN',
       'route_id_direction_8_TO DOWNTOWN', 'route_id_direction_93_LOOP',
       'route_id_direction_94_FROM NASHVILLE',
       'route_id_direction_95_FROM NASHVILLE',
       'route_id_direction_96_FROM NASHVILLE',
       'route_id_direction_96_TO NASHVILLE',
       'route_id_direction_9_FROM DOWNTOWN',
       'route_id_direction_9_TO DOWNTOWN', 'is_holiday_False',
       'is_holiday_True', 'dayofweek_1', 'dayofweek_2', 'dayofweek_3',
       'dayofweek_4', 'dayofweek_5', 'dayofweek_6', 'dayofweek_7', 'year_ix',
       'month_ix', 'hour_ix', 'day_ix']
    
    ix_map = joblib.load('data/mta_day_ahead/TL_IX_map.joblib')
    ohe_encoder = joblib.load('data/mta_day_ahead/TL_OHE_encoders.joblib')
    
    # OHE for route_id_direction
    input_df[ohe_encoder.get_feature_names_out()] = ohe_encoder.transform(input_df[cat_features]).toarray()
    input_df = input_df.drop(columns=cat_features)
    
    # label encode of categorical variables
    for col in ord_features:
        input_df[f'{col}_ix'] = input_df[col].apply(lambda x: ix_map[col][x])
    input_df = input_df.drop(columns=ord_features)
    
    input_df = input_df[train_columns]
    input_df = input_df.dropna()
    input_df[train_columns[0:5]] = input_df[train_columns[0:5]].apply(pd.to_numeric)
    return input_df

def load_weather_data(path='data/mta_day_ahead/darksky.nashville.csv'):
    darksky = pd.read_csv(path)
    # GMT-5
    darksky['datetime'] = darksky['time'] - 18000
    darksky['datetime'] = pd.to_datetime(darksky['datetime'], infer_datetime_format=True, unit='s')
    darksky = darksky.set_index(darksky['datetime'])
    darksky['year'] = darksky['datetime'].dt.year
    darksky['month'] = darksky['datetime'].dt.month
    darksky['day'] = darksky['datetime'].dt.day
    darksky['hour'] = darksky['datetime'].dt.hour
    val_cols= ['temperature', 'humidity', 'precipitation_intensity']
    join_cols = ['year', 'month', 'day', 'hour']
    darksky = darksky[val_cols+join_cols]
    renamed_cols = {k: f"darksky_{k}" for k in val_cols}
    darksky = darksky.rename(columns=renamed_cols)
    darksky = darksky.groupby(['year', 'month', 'day', 'hour']).mean().reset_index()
    return darksky

def load_holiday_data(path='data/mta_day_ahead/US Holiday Dates (2004-2021).csv'):
    # Holidays
    holidays_df = pd.read_csv(path)
    additional_df = []
    additional_df.append(pd.Series({'Date': '2022-01-01', 'Holiday': "New Year's Day"}).to_frame().T)
    additional_df.append(pd.Series({'Date': '2022-01-17', 'Holiday': "Martin Luther King, Jr. Day"}).to_frame().T)
    additional_df.append(pd.Series({'Date': '2022-04-15', 'Holiday': "Good Friday"}).to_frame().T)
    additional_df.append(pd.Series({'Date': '2022-05-30', 'Holiday': "Memorial Day"}).to_frame().T)
    holidays_df = pd.concat([holidays_df] + additional_df)
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])
    return holidays_df


def setup_day_ahead_data(DATE_TO_PREDICT, past_df, darksky, holidays_df, TARGET='load'):
    DAY_OF_WEEK = convert_pandas_dow_to_pyspark(pd.Timestamp(DATE_TO_PREDICT).day_of_week)
    a = past_df.groupby(['route_id_direction', 'time_window']).agg({'actual_headways':list, 'scheduled_headway': list, TARGET: list})
    a['hour'] = a.index.get_level_values('time_window') // 2
    d = darksky[['hour', 'darksky_temperature', 'darksky_humidity', 'darksky_precipitation_intensity']]
    d = darksky[(darksky['year']==pd.Timestamp(DATE_TO_PREDICT).year) & 
                (darksky['month']==pd.Timestamp(DATE_TO_PREDICT).month) & 
                (darksky['day']==pd.Timestamp(DATE_TO_PREDICT).day)][['hour', 'darksky_temperature', 'darksky_humidity', 'darksky_precipitation_intensity']]

    a = a.reset_index()
    a = a.merge(d, left_on='hour', right_on='hour').sort_values(by=['route_id_direction', 'time_window'])
    a['arrival_time'] = a['time_window'].apply(lambda x: pd.Timestamp(f"{pd.Timestamp(DATE_TO_PREDICT) + pd.Timedelta(str(x * 30) + 'min')}"))
    a['transit_date'] = DATE_TO_PREDICT
    a['transit_date'] = pd.to_datetime(a['transit_date'])
    a['year'] = pd.Timestamp(DATE_TO_PREDICT).year
    a['month'] = pd.Timestamp(DATE_TO_PREDICT).month
    a['day'] = pd.Timestamp(DATE_TO_PREDICT).day
    a['dayofweek'] = DAY_OF_WEEK
    a['is_holiday'] = not holidays_df[holidays_df['Date'] == pd.Timestamp(DATE_TO_PREDICT)].empty
    # a['sched_hdwy95'] = a['scheduled_headway'].apply(lambda x: np.percentile(x, 95, interpolation='lower'))
    a['sched_hdwy95'] = a['scheduled_headway'].apply(lambda x: np.max(x))

    a = a.drop(['actual_headways', 'scheduled_headway', TARGET], axis=1)
    a = a.rename({'darksky_temperature':'temperature', 'darksky_humidity':'humidity', 'darksky_precipitation_intensity': 'precipitation_intensity',
                'sched_hdwy95':'scheduled_headway'}, axis=1)
    a = a.bfill()
    return a

def setup_input_data(DATE_TO_PREDICT, past_df):
    darksky = load_weather_data()
    holidays_df = load_holiday_data()
    input_df = setup_day_ahead_data(DATE_TO_PREDICT, past_df, darksky, holidays_df)
    return input_df

def generate_results(input_df):
    results = input_df.groupby('route_id_direction').agg({'y_pred': list, 'time_window': list})
    a = pd.DataFrame(columns=list(range(0, 48)))
    for i, (k, v) in enumerate(results.iterrows()):
        a.loc[i, v['time_window']] = v['y_pred']
    a['route'] = results.index
    a.index = a['route']
    a = a.drop('route', axis=1)
    a = a.apply(pd.to_numeric, errors='coerce')
    a.columns = a.columns.astype('int')
    return a


def create_random_data(DATE_TO_PREDICT):
    all_routes_list = pd.read_csv("data/mta_day_ahead/all_routes.txt")
    random_routes = all_routes_list.sample(frac=round(np.random.uniform(0.1, 1.0), 10))['0'].tolist()
    time_windows = [random.sample(range(10, 47), random.randint(0, 47-10)) for _ in random_routes]
    is_holiday = np.random.choice(a=[False, True])
    date = DATE_TO_PREDICT
    
                    # 'avg_past_act_headway',
                    # # schedheadway
                    # 'avg_past_trips_loads',
                    # 'act_headway_pct_change', 
                    # 'load_pct_change', 
    column_names = ['route_id_direction', 'time_window', 'hour',
                    'temperature', 'humidity', 'precipitation_intensity',
                    'arrival_time', 'transit_date', 'year', 'is_holiday',
                    'scheduled_headway', 
                    'month', 'day', 'dayofweek']
    input_df = []
    for i, random_route in enumerate(random_routes):
        r = np.zeros((len(time_windows[i]), 11))
        r[:,1] = time_windows[i]
        
        r[:,3] = np.random.uniform(40.0, 100.0)
        r[:,4:6] = np.random.rand(len(time_windows[i]), 2)
        r[:,9] = is_holiday
        r[:,10] = np.random.uniform(2000.0, 3600.0, len(time_windows[i]))
        a = pd.DataFrame(r)
        a[0] = random_route
        a[2] = a[1]//2
        a[7] = date
        a[6] = a[1].apply(lambda x: pd.Timestamp(f"{pd.Timestamp(date) + pd.Timedelta(str(x * 30) + 'min')}"))
        a[8] = pd.Timestamp(date).year
        a['month'] = pd.Timestamp(date).month
        a['day'] = pd.Timestamp(date).day
        a['dayofweek'] = pd.Timestamp(date).day_of_week
        a[0] = random_route
        print(a.shape)
        a.columns = column_names
        input_df.append(a)
    input_df = pd.concat(input_df)
    return input_df