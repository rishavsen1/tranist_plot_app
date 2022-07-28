from numpy import percentile
import streamlit as st
import datetime as dt
import os
from src import config, stop_level_utils
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff
import pandas as pd

st.markdown("# Stop level Prediction")
st.sidebar.markdown("# Parameters")

def get_data(predict_date, predict_time, minute_range=10):
    f = os.path.join('data', config.MTA_PARQUET)
    apcdata = spark.read.load(f)
    apcdata.createOrReplaceTempView("apc")
    get_columns = ['trip_id', 'transit_date', 'arrival_time', 
                   'stop_sequence',
                   'load', 
                   'darksky_temperature', 
                   'darksky_humidity', 
                   'darksky_precipitation_probability', 
                   'route_direction_name', 'route_id',
                   'dayofweek',  'year', 'month', 'hour',
                   'sched_hdwy']
    get_str = ", ".join([c for c in get_columns])

    query = f"""
        SELECT {get_str}
        FROM apc
        WHERE (transit_date == '{predict_date.strftime('%Y-%m-%d')}')
        """
    apcdata = spark.sql(query)
    apcdata = apcdata.na.drop(subset=["arrival_time", "darksky_temperature"])
    # TODO: Fix time zone issues
    start = dt.datetime.combine(predict_date, predict_time) - dt.timedelta(minutes=minute_range) - dt.timedelta(hours=5)
    end   = dt.datetime.combine(predict_date, predict_time) + dt.timedelta(minutes=minute_range) - dt.timedelta(hours=5)
    apcdata = apcdata.filter(F.col("arrival_time").between(start, end))
    apcdata = apcdata.withColumn("route_id_dir", F.concat_ws("_", apcdata.route_id, apcdata.route_direction_name))
    apcdata = apcdata.withColumn("day", F.dayofmonth(apcdata.arrival_time))
    apcdata = apcdata.drop("route_id","route_direction_name")
    apcdata = apcdata.withColumn("load", F.when(apcdata.load < 0, 0).otherwise(apcdata.load))
    df = apcdata.toPandas()
    
    if df.empty:
        st.error("Input data does not exist.")
    df = df.groupby(['route_id_dir', 'stop_sequence']).agg({'trip_id':'first',
                                                            'transit_date':'first',
                                                            'arrival_time':'first',
                                                            'load':'max',
                                                            'darksky_temperature':'max',
                                                            'darksky_humidity':'max',
                                                            'darksky_precipitation_probability':'max',
                                                            'dayofweek':'first','year':'first','month':'first','day':'first','hour':'first',
                                                            'sched_hdwy':'max'}).reset_index()
    percentiles = [(0.0, 6.0), (7.0, 12.0), (13.0, 100.0)]
    df[config.TARGET_COLUMN_CLASSIFICATION] = df['load'].apply(lambda x: stop_level_utils.get_class(x, percentiles))
    return df

with st.sidebar:
    dataset_selectbox = st.selectbox('Dataset', ('Nashville, MTA', 'Chattanooga, CARTA'))
    
    if dataset_selectbox == 'Nashville, MTA':
        predict_date = st.date_input('Date to predict:', min_value=dt.date(2021, 10, 31), max_value=dt.date(2022, 4, 5), value=dt.date(2022, 1, 5))
    elif dataset_selectbox == 'Chattanooga, CARTA':
        predict_date = st.date_input('Date to predict:', min_value=dt.date(2022, 5, 30), value=dt.date(2022, 6, 1))
    
    predict_time = st.time_input('Time to predict:', dt.time(8, 45))
    predict_button = st.button('Plot predictions')
    
spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '80g')\
    .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '40g').master("local[26]")\
    .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
    .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
    .config("spark.sql.autoBroadcastJoinThreshold", -1)\
    .config("spark.driver.maxResultSize", 0)\
    .config("spark.shuffle.spill", "true")\
    .getOrCreate()
    
if predict_button:
    with st.spinner("Loading..."):
        minute_range = 15
        start = dt.datetime.combine(predict_date, predict_time) - dt.timedelta(minutes=minute_range)
        end   = dt.datetime.combine(predict_date, predict_time) + dt.timedelta(minutes=minute_range)
        input_df = get_data(predict_date, predict_time, minute_range)
        st.write("Input data")
        st.dataframe(input_df)
        keep_columns=['route_id_dir', 'trip_id']
        input_df = stop_level_utils.prepare_input_data(input_df, keep_columns=keep_columns)
        
        num_features = input_df.shape[1] - len(keep_columns)
        model = stop_level_utils.get_model(num_features)
        past = 5
        
        results_df = []
        for i, route_df in input_df.groupby(['route_id_dir']):
            route_df = route_df.drop(columns=keep_columns)
            if len(route_df) <= past:
                continue
            future = min(10, len(route_df) - past)
            y_past = route_df.iloc[:past]['y_class'].tolist()
            y_true = route_df.iloc[past:future]['y_class'].tolist()
            y_pred = stop_level_utils.generate_simple_lstm_predictions(route_df, model, past, future)
            route_df['y_pred'] = y_past + y_pred + [np.nan]*(len(route_df) - future - past)
            route_df['type'] = ['past']*len(y_past) + ['pred']*len(y_pred) + [np.nan]*(len(route_df) - future - past)
            route_df['route_id_dir'] = i
            results_df.append(route_df)
            
        results_df = pd.concat(results_df)
        results_df = results_df.dropna(subset=['y_pred'])
        
        fig = px.scatter(results_df, x="stop_sequence", y="y_pred", facet_col="route_id_dir", facet_col_wrap=4, color='type',
                    facet_row_spacing=0.07, # default is 0.07 when facet_col_wrap is used
                    facet_col_spacing=0.08, # default is 0.03,
                    height=1000, width=800)
        
        fig.update_xaxes(title="Stop ID")
        fig.update_yaxes(title="Occupancy")
        layout = fig.update_layout(
            title=f'Stop level occupancy for trips between {start} and {end}',
        )
        
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_yaxes(showticklabels=True)
        st.plotly_chart(fig)