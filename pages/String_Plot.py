import streamlit as st
import datetime as dt
import os
from src import config, plot_utils
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
import plotly.graph_objects as go
from copy import deepcopy
import numpy as np

spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '40g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '20g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .getOrCreate()

def get_route_list(plot_date):
    filepath = os.path.join(os.getcwd(), "data", config.MTA_PARQUET)
    apcdata = spark.read.load(filepath)
    apcdata.createOrReplaceTempView("apc")

    plot_date = filter_date.strftime('%Y-%m-%d')
    # filter subset
    query = f"""
            SELECT transit_date, route_id
            FROM apc
            WHERE (transit_date == '{plot_date}')
            """
    apcdata = spark.sql(query)
    apcdata = apcdata.dropDuplicates(["route_id"])
    apcdata = apcdata.sort(apcdata.route_id)
    return list(apcdata.select('route_id').toPandas()['route_id'])

def get_vehicle_list(route_option):
    filepath = os.path.join(os.getcwd(), "data", config.MTA_PARQUET)
    apcdata = spark.read.load(filepath)
    apcdata.createOrReplaceTempView("apc")

    plot_date = filter_date.strftime('%Y-%m-%d')
    # filter subset
    query = f"""
            SELECT vehicle_id, transit_date, route_id
            FROM apc
            WHERE (transit_date == '{plot_date}')
            """
    apcdata = spark.sql(query)
    apcdata = apcdata.where(apcdata.route_id == route_option)
    apcdata = apcdata.dropDuplicates(["vehicle_id"])
    apcdata = apcdata.sort(apcdata.vehicle_id)
    return list(apcdata.select('vehicle_id').toPandas()['vehicle_id'])

def assign_data_to_bins(df, data_option):
    percentiles = [0, 33, 66, 100]
    if data_option == 'Boardings':
        bins = [-1, 6, 10, 15, 100]
        labels = [0, 1, 2, 3]
        df['y_class'] = pd.cut(df['ons'], bins=bins, labels=labels)
    return df

st.markdown("# String Plots")
st.sidebar.markdown("# Parameters")

with st.sidebar:
    dataset_selectbox = st.selectbox('Dataset', ('Nashville, MTA', 'Chattanooga, CARTA'))
    if dataset_selectbox == 'Chattanooga, CARTA':
        fp = os.path.join('data', 'CARTA_route_ids.csv')
        filter_date = st.date_input('Filter dates', 
                                    min_value=dt.date(2019, 1, 1), max_value=dt.date(2022, 5, 30),
                                    value=dt.date(2021, 10, 18))
    elif dataset_selectbox == 'Nashville, MTA':
        filter_date = st.date_input('Filter dates', 
                                    min_value=dt.date(2020, 1, 1), max_value=dt.date(2022, 4, 6),
                                    value=dt.date(2021, 10, 18))

        route_list = get_route_list(filter_date)
        route_option = st.selectbox('Route:', route_list)
        vehicle_list = get_vehicle_list(route_option)
        vehicle_options = st.multiselect('Vehicles:', vehicle_list)
        vehicle_options = [str(v) for v in vehicle_options]
        
        data_options = st.selectbox('Data to show:', ['Boardings', 'Occupancy'])
        if data_options == 'Occupancy':
            predict_time = st.time_input('Time to predict:', dt.time(8, 45))
    plot_button = st.button('Plot graphs')
    
if plot_button:
    if dataset_selectbox == 'Chattanooga, CARTA':
        st.error("Not yet prepared.")
    else:
        with st.spinner("Loading..."):
            filepath = os.path.join(os.getcwd(), "data", config.MTA_PARQUET)
            apcdata = spark.read.load(filepath)
            apcdata.createOrReplaceTempView("apc")

            plot_date = filter_date.strftime('%Y-%m-%d')
            # filter subset
            query = f"""
                    SELECT trip_id, gtfs_direction_id, arrival_time, transit_date, route_id, vehicle_id, ons, block_abbr,
                        stop_name, stop_id_original, stop_sequence
                    FROM apc
                    WHERE (transit_date == '{plot_date}')
                    ORDER BY arrival_time
                    """
            apcdata = spark.sql(query)
            apcdata = apcdata.where(apcdata.route_id == route_option)
            apcdata = apcdata.where(F.col("vehicle_id").isin(vehicle_options))
            df = apcdata.toPandas()
            df = assign_data_to_bins(df, data_options)
            if data_options == 'Boardings':
                fig = plot_utils.plot_string_boarding(df, vehicle_options)
            else:
                fig = plot_utils.plot_string_occupancy(df, filter_date, vehicle_options, predict_time)
            st.plotly_chart(fig)