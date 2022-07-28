# Contents of ~/my_app/streamlit_app.py
import streamlit as st
import datetime as dt
import os
from src import plot_utils
from src import data_utils
from src import day_ahead_utils
from src import config
from pyspark.sql import SparkSession
import pandas as pd
import joblib

format_dict = {"time_window":"{:.1f}", "hour":"{:.0f}", 
                "temperature":"{:.2f}",
                "humidity":"{:.2f}",
                "precipitation_intensity":"{:.2f}",
                "scheduled_headway":"{:.2f}"}

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

st.markdown("# Day Ahead Prediction")
st.sidebar.markdown("# Parameters")

with st.sidebar:
    dataset_selectbox = st.selectbox('Dataset', ('Nashville, MTA', 'Chattanooga, CARTA'))
    
    if dataset_selectbox == 'Nashville, MTA':
        predict_date = st.date_input('Date to predict:', min_value=dt.date(2022, 4, 7), value=dt.date(2022, 4, 7))
    elif dataset_selectbox == 'Chattanooga, CARTA':
        predict_date = st.date_input('Date to predict:', min_value=dt.date(2022, 5, 30), value=dt.date(2022, 6, 1))
        
    random_df = st.radio(
        "Data generation:",
        ('from past', 'random'),
        horizontal=True)
    routes = st.radio(
        "Routes:",
        ('only active routes', 'all routes'),
        horizontal=True)
    
    predict_button = st.button('Plot predictions')

spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '40g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '20g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .getOrCreate()

agg_time = 30
dates = pd.date_range(pd.Timestamp(predict_date), pd.Timestamp(predict_date) + pd.Timedelta('24h') - pd.Timedelta(f'{agg_time}m'), freq=f'{agg_time}min')

if predict_button:
    with st.spinner(f"Loading models..."):
        model012    = joblib.load('data/mta_day_ahead/XGB_012_NoPastInfo_30min.joblib')
        model234    = joblib.load('data/mta_day_ahead/XGB_234_NoPastInfo_30min.joblib')
        columns     = joblib.load('data/mta_day_ahead/TL_X_columns.joblib')
        ohe_encoder = joblib.load('data/mta_day_ahead/TL_OHE_encoders.joblib')
    
    with st.spinner(f"Generating data to predict..."):
        if random_df == 'random':
            input_df = day_ahead_utils.create_random_data(predict_date)
            st.dataframe(input_df.style.format(format_dict))
            input_df = day_ahead_utils.prepare_day_ahead_for_prediction(input_df)
        else:
            past_df = day_ahead_utils.get_past_data(spark, predict_date)
            input_df = day_ahead_utils.setup_input_data(predict_date, past_df)
            st.dataframe(input_df.style.format(format_dict))
            input_df = day_ahead_utils.prepare_day_ahead_for_prediction(input_df)

    with st.spinner(f"Predicting..."):
        # Prediction
        input_df = input_df.reset_index(drop=True)
        input_df = input_df[columns]
        
        ## Predict first stage 0-1-2
        predictions = model012.predict(input_df)
        input_df['y_pred'] = predictions
        ## Isolate predictions with bin 2 for 2-3-4
        high_bin_df = input_df[input_df['y_pred'] == 2]
        high_bin_df = high_bin_df.drop(['y_pred'], axis=1)
        high_bin_index = high_bin_df.index            
        predictions = model234.predict(high_bin_df)
        predictions = predictions + 2
        input_df.loc[high_bin_index, 'y_pred'] = predictions
        ohe_features = ['route_id_direction', 'is_holiday', 'dayofweek']
        input_df[ohe_features] = ohe_encoder.inverse_transform(input_df.filter(regex='route_id_direction_|is_holiday_|dayofweek_'))
        df = generate_results(input_df)
        if routes == 'all routes':                
            all_routes_list = pd.read_csv("data/mta_day_ahead/all_routes.txt")['0'].tolist()
            new_index = pd.Index(all_routes_list, name="route_id_direction")
            df = df.reset_index()
            df = df.set_index("route").reindex(new_index).reset_index()
            df = df.set_index('route_id_direction')
            df = df.sort_index()
            
        ## Plotting
        fig = plot_utils.plot_prediction_heatmap(predict_date, df)
        st.plotly_chart(fig, use_container_width=True)
        
    # if dataset_selectbox == 'Nashville, MTA':
    #     filepath = os.path.join(os.getcwd(), "data", config.MTA_PARQUET)
    # elif dataset_selectbox == 'Chattanooga, CARTA':
    #     filepath = os.path.join(os.getcwd(), "data", config.CARTA_PARQUET)
    # else:
    #     st.error("Select dataset")
        
    # with st.spinner(f"Loading {dataset_selectbox} files..."):
    #     apcdata = spark.read.load(filepath)
    #     apcdata.createOrReplaceTempView("apc")
        
    #     # filter subset
    #     query = f"""
    #             SELECT *
    #             FROM apc
    #             WHERE (transit_date >= '{start}') AND (transit_date <= '{end}')
    #             """
        
    #     if dataset_selectbox == 'Nashville, MTA':
    #         apcdata=spark.sql(query)
    #         apcdata = data_utils.remove_nulls_from_apc(apcdata)
    #         apcdata.createOrReplaceTempView('apcdata')
    #         apcdata_per_trip = data_utils.get_apc_per_trip_sparkview(spark)
    #         apcdata_per_trip = apcdata_per_trip.withColumnRenamed("route_id_direction","route_id_dir")
    #         apcdata_per_trip = apcdata_per_trip.drop("load")
    #         df = apcdata_per_trip.toPandas()
    #     elif dataset_selectbox == 'Chattanooga, CARTA':
    #         apcdata_per_trip=spark.sql(query)
    #         apcdata_per_trip = apcdata_per_trip.na.drop(subset=["time_actual_arrive"])
    #         df = apcdata_per_trip.toPandas()

    #     st.dataframe(df.head())
    #     st.dataframe(df.tail())
        
    # with st.spinner(f"Preparing {dataset_selectbox} graphs..."):
    #     fig = plot_utils.plot_max_aggregate(df, filter_date[0], None, time_granularity, dataset_selectbox)
    #     st.plotly_chart(fig, use_container_width=True)