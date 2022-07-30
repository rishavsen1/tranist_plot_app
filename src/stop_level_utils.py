import joblib
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow import keras
import numpy as np
import pandas as pd
from copy import deepcopy
import datetime as dt

def get_class(val, percentiles):
    for i, (min, max) in enumerate(percentiles):
        if (val >= min) and (val <= max):
            return i
    return None

# LSTM
def prepare_input_data(input_df, keep_columns=[], target='y_class'):
    num_columns = ['darksky_temperature', 'darksky_humidity', 'darksky_precipitation_probability', 'sched_hdwy']
    cat_columns = ['month', 'hour', 'day', 'stop_sequence', 'year']
    ohe_columns = ['dayofweek', 'route_id_dir']

    columns = ['y_class', 'stop_sequence', 'darksky_temperature', 'darksky_humidity', 'darksky_precipitation_probability', 
            'year', 'month', 'hour', 'sched_hdwy', 'day', 'dayofweek_1', 'dayofweek_2', 'dayofweek_3', 'dayofweek_4', 'dayofweek_5', 'dayofweek_6', 'dayofweek_7',
            'route_id_dir_14_FROM DOWNTOWN', 'route_id_dir_14_TO DOWNTOWN', 'route_id_dir_17_FROM DOWNTOWN', 'route_id_dir_17_TO DOWNTOWN',
            'route_id_dir_18_FROM DOWNTOWN', 'route_id_dir_18_TO DOWNTOWN', 'route_id_dir_19_FROM DOWNTOWN', 'route_id_dir_19_TO DOWNTOWN',
            'route_id_dir_21_NORTHBOUND', 'route_id_dir_21_SOUTHBOUND', 'route_id_dir_22_FROM DOWNTOWN', 'route_id_dir_22_TO DOWNTOWN',
            'route_id_dir_23_FROM DOWNTOWN', 'route_id_dir_23_TO DOWNTOWN', 'route_id_dir_24_FROM DOWNTOWN', 'route_id_dir_24_TO DOWNTOWN',
            'route_id_dir_25_NORTHBOUND', 'route_id_dir_25_SOUTHBOUND', 'route_id_dir_28_FROM DOWNTOWN', 'route_id_dir_28_TO DOWNTOWN',
            'route_id_dir_29_FROM DOWNTOWN', 'route_id_dir_29_TO DOWNTOWN', 'route_id_dir_34_FROM DOWNTOWN', 'route_id_dir_34_TO DOWNTOWN',
            'route_id_dir_35_FROM DOWNTOWN', 'route_id_dir_35_TO DOWNTOWN', 'route_id_dir_38_FROM DOWNTOWN', 'route_id_dir_38_TO DOWNTOWN',
            'route_id_dir_3_FROM DOWNTOWN', 'route_id_dir_3_TO DOWNTOWN', 'route_id_dir_41_FROM DOWNTOWN', 'route_id_dir_41_TO DOWNTOWN',
            'route_id_dir_42_FROM DOWNTOWN', 'route_id_dir_42_TO DOWNTOWN', 'route_id_dir_43_FROM DOWNTOWN', 'route_id_dir_43_TO DOWNTOWN',
            'route_id_dir_4_FROM DOWNTOWN', 'route_id_dir_4_TO DOWNTOWN', 'route_id_dir_50_FROM DOWNTOWN', 'route_id_dir_50_TO DOWNTOWN',
            'route_id_dir_52_FROM DOWNTOWN', 'route_id_dir_52_TO DOWNTOWN', 'route_id_dir_55_FROM DOWNTOWN', 'route_id_dir_55_TO DOWNTOWN',
            'route_id_dir_56_FROM DOWNTOWN', 'route_id_dir_56_TO DOWNTOWN', 'route_id_dir_5_FROM DOWNTOWN', 'route_id_dir_5_TO DOWNTOWN',
            'route_id_dir_64_FROM RIVERFRONT', 'route_id_dir_64_TO RIVERFRONT', 'route_id_dir_6_FROM DOWNTOWN', 'route_id_dir_6_TO DOWNTOWN',
            'route_id_dir_72_EDMONDSON', 'route_id_dir_72_GRASSMERE', 'route_id_dir_75_NORTHBOUND', 'route_id_dir_75_SOUTHBOUND', 'route_id_dir_76_LOOP',
            'route_id_dir_79_EASTBOUND', 'route_id_dir_79_NORTHBOUND', 'route_id_dir_7_FROM DOWNTOWN', 'route_id_dir_7_TO DOWNTOWN',
            'route_id_dir_84_FROM NASHVILLE', 'route_id_dir_84_TO NASHVILLE', 'route_id_dir_86_FROM NASHVILLE', 'route_id_dir_86_TO NASHVILLE',
            'route_id_dir_8_FROM DOWNTOWN', 'route_id_dir_8_TO DOWNTOWN', 'route_id_dir_93_LOOP', 'route_id_dir_94_FROM NASHVILLE',
            'route_id_dir_95_FROM NASHVILLE', 'route_id_dir_96_FROM NASHVILLE', 'route_id_dir_96_TO NASHVILLE', 'route_id_dir_9_FROM DOWNTOWN', 'route_id_dir_9_TO DOWNTOWN']
    
    label_encoders = joblib.load('data/mta_stop_level/LL_Label_encoders.joblib')
    ohe_encoder = joblib.load('data/mta_stop_level/LL_OHE_encoders.joblib')
    num_scaler = joblib.load('data/mta_stop_level/LL_Num_scaler.joblib')
    
    # OHE
    input_df[ohe_encoder.get_feature_names_out()] = ohe_encoder.transform(input_df[ohe_columns]).toarray()
    # input_df = input_df.drop(columns=ohe_columns)

    # Label encoder
    for cat in cat_columns:
        encoder = label_encoders[cat]
        input_df[cat] = encoder.transform(input_df[cat])
    
    # Num scaler
    input_df[num_columns] = num_scaler.transform(input_df[num_columns])
    input_df['y_class']  = input_df.y_class.astype('int')

    if keep_columns:
        columns = keep_columns + columns
    # Rearrange columns
    input_df = input_df[columns]
    
    return input_df

def setup_simple_lstm_generator(num_features, num_classes, learning_rate=1e-4):
    # define model
    model = tf.keras.Sequential()
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # compile model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["sparse_categorical_accuracy"],
    )

    input_shape = (None, None, num_features)
    model.build(input_shape)
    return model

def get_model(num_features):
    simple_lstm = setup_simple_lstm_generator(num_features, 3)
    # Load model
    latest = tf.train.latest_checkpoint('data/mta_stop_level')
    simple_lstm.load_weights(latest)
    return simple_lstm

def generate_simple_lstm_predictions(input_df, model, past, future):
    past_df = input_df[0:past]
    future_df = input_df[past:]
    predictions = []
    for f in range(future):
        pred = model.predict(past_df.to_numpy().reshape(1, *past_df.shape))
        y_pred = np.argmax(pred)
        predictions.append(y_pred)
        
        # Add information from future
        last_row = future_df.iloc[[0]]
        last_row['y_class'] = y_pred
        past_df = pd.concat([past_df[1:], last_row])
        
        # Move future to remove used row
        future_df = future_df[1:]
    return predictions

# route 3 vehicle 1830
def setup_past_future_from_datetime(df, filter_datetime, past=5, future=10):
    # time_now = dt.time(16, 35)
    # For getting some time to base stop level prediction
    # time_now = dt.datetime.now().time()
    # datetime_now = dt.datetime.combine(filter_date, time_now)
    tdf = deepcopy(df).dropna().reset_index()
    tdf = tdf.sort_values('arrival_time').drop_duplicates('arrival_time',keep='last')
    tdf.index = tdf['arrival_time']
    idx = tdf.iloc[tdf.index.get_indexer([filter_datetime], method='nearest')]
    past_idx = tdf.loc[idx.index]['index'].values[0]
    print(past_idx)
    if past_idx < past:
        return pd.DataFrame(), pd.DataFrame()
    return df.loc[past_idx - past:past_idx], df.loc[past_idx+1:past_idx+1 + future]