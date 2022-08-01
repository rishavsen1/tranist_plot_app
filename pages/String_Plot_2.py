import streamlit as st
import datetime as dt
import os
from src import config
from src import plot_utils_2 as plot_utils
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd

spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '40g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '20g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .getOrCreate()

def get_block_abbr_list(apcdata):
    apcdata = apcdata.dropDuplicates(["block_abbr"])
    apcdata = apcdata.sort(apcdata.block_abbr)
    return [r[0] for r in apcdata.select('block_abbr').distinct().toLocalIterator()]
    
def get_route_list(apcdata):
    apcdata = apcdata.dropDuplicates(["route_id"])
    apcdata = apcdata.sort(apcdata.route_id)
    return list(apcdata.select('route_id').toPandas()['route_id'])

def get_vehicle_list(apcdata, route_option):
    apcdata = apcdata.where(apcdata.route_id == route_option)
    apcdata = apcdata.dropDuplicates(["vehicle_id"])
    apcdata = apcdata.sort(apcdata.vehicle_id)
    return list(apcdata.select('vehicle_id').toPandas()['vehicle_id'])

def assign_data_to_bins(df, data_option):
    if data_option == 'Boardings':
        bins = pd.IntervalIndex.from_tuples([(-1, 5), (5, 11), (11, 16), (16, 29), (29, 101)])
        mycut = pd.cut(df['ons'].tolist(), bins=bins)
    if data_option == 'Occupancy':
        bins = pd.IntervalIndex.from_tuples([(-1, 6), (6, 12), (12, 100)])
        mycut = pd.cut(df['ons'].tolist(), bins=bins)
        
    df['y_class'] = mycut.codes
    return df

def get_apc_data_for_date(filter_date):
    filepath = os.path.join(os.getcwd(), "data", config.MTA_PARQUET)
    apcdata = spark.read.load(filepath)
    apcdata.createOrReplaceTempView("apc")

    plot_date = filter_date.strftime('%Y-%m-%d')    
    get_columns = ['trip_id', 'transit_date', 'arrival_time', 'vehicle_id', 'ons',
                   'block_abbr', 'stop_sequence', 'stop_name', 'stop_id_original',
                   'load', 
                   'darksky_temperature', 
                   'darksky_humidity', 
                   'darksky_precipitation_probability', 
                   'route_direction_name', 'route_id', 'gtfs_direction_id',
                   'dayofweek',  'year', 'month', 'hour',
                   'sched_hdwy']
    get_str = ", ".join([c for c in get_columns])
    query = f"""
    SELECT {get_str}
    FROM apc
    WHERE (transit_date == '{plot_date}')
    ORDER BY arrival_time
    """
    apcdata = spark.sql(query)
    apcdata = apcdata.withColumn("route_id_dir", F.concat_ws("_", apcdata.route_id, apcdata.route_direction_name))
    apcdata = apcdata.withColumn("day", F.dayofmonth(apcdata.arrival_time))
    apcdata = apcdata.drop("route_direction_name")
    apcdata = apcdata.withColumn("load", F.when(apcdata.load < 0, 0).otherwise(apcdata.load))
    return apcdata
    
st.title("String Plots")
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
        apcdata = get_apc_data_for_date(filter_date)
        block_abbr_list = get_block_abbr_list(apcdata)
        block_option = st.selectbox('Block:', block_abbr_list)

    data_options = st.selectbox('Data to show:', ['Boardings', 'Occupancy'])
    if data_options == 'Occupancy':
        predict_time = st.time_input('Time to predict:', dt.time(12, 45))
container = st.container()

col1, col2 = container.columns(2)
with col1:
    st.subheader("First Y-Axis")
    enable1 = st.checkbox('Enable1?', True)
    if enable1:
        route_list1 = get_route_list(apcdata)
        route_option1 = st.selectbox('Route1:', route_list1)
        direction_option_label1 = st.radio("Direction1:", ["To Downtown", "From Downtown"], horizontal=True)
        direction_option1 = 0 if direction_option_label1 == "To Downtown" else 1

        container_in1 = st.container()
        all1 = st.checkbox("Select All1")
        vehicle_list1 = get_vehicle_list(apcdata, route_option1)
        if all1:
            vehicle_options1 = container_in1.multiselect("Vehicles:", vehicle_list1, vehicle_list1)
        else:
            vehicle_options1 =  container_in1.multiselect("Vehicles:", vehicle_list1)
        
with col2:
    st.subheader("Second Y-Axis")
    enable2 = st.checkbox('Enable2?', True)
    if enable2:
        route_list2 = get_route_list(apcdata)
        route_option2 = st.selectbox('Route2:', route_list2)
        direction_option_label2 = st.radio("Direction2:", ["From Downtown", "To Downtown"], horizontal=True)
        direction_option2 = 0 if direction_option_label2 == "To Downtown" else 1

        container_in2 = st.container()
        all2 = st.checkbox("Select All2")
        vehicle_list2 = get_vehicle_list(apcdata, route_option2)
        if all2:
            vehicle_options2 = container_in2.multiselect("Vehicles2:", vehicle_list2, vehicle_list2)
        else:
            vehicle_options2 =  container_in2.multiselect("Vehicles2:", vehicle_list2)
    
plot_button = st.button('Plot graphs')
    
if plot_button:
    with st.spinner("Processing..."):
        # First graph
        if enable1:
            apcdata1 = apcdata.where(apcdata.route_id == route_option1)
            apcdata1 = apcdata1.where(F.col("vehicle_id").isin(vehicle_options1))
            apcdata1 = apcdata1.where(apcdata1.gtfs_direction_id == direction_option1)
            if apcdata1.count() == 0:
                st.error("First plot is empty.")
            df1 = apcdata1.toPandas()
            df1['plot_no'] = 0
        
        # Second graph
        if enable2:
            apcdata2 = apcdata.where(apcdata.route_id == route_option2)
            apcdata2 = apcdata2.where(F.col("vehicle_id").isin(vehicle_options2))
            apcdata2 = apcdata2.where(apcdata2.gtfs_direction_id == direction_option2)
            if apcdata2.count() == 0:
                st.error("Second plot is empty.")
            df2 = apcdata2.toPandas()
            df2['plot_no'] = 1
        
        if enable1 and enable2:
            if direction_option1 == direction_option2:
                df = df1
            else:
                df = pd.concat([df1, df2])
            vehicle_options = list(set(vehicle_options1 + vehicle_options2))
        elif enable1 and not enable2:
            df = df1
            vehicle_options = vehicle_options1
        elif not enable1 and enable2:
            df = df2
            vehicle_options = vehicle_options2
        df = assign_data_to_bins(df, data_options)
        
        if data_options == 'Boardings':
            fig = plot_utils.plot_string_boarding(df)
        else:
            fig = plot_utils.plot_string_occupancy(df, filter_date, predict_time)
            
    fig.update_layout(title=f'{data_options} for Block: {block_option}, Route: {route_option1} {direction_option_label1} and {route_option2} {direction_option_label2}')
    st.plotly_chart(fig)