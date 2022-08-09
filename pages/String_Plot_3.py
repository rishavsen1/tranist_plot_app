import streamlit as st
import datetime as dt
import os
from src import config
from src import plot_utils_2 as plot_utils
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
import plotly.express as px
from shapely.geometry import Point
import geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import numpy as np

spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '40g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '20g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .getOrCreate()

def get_block_abbr_list(apcdata, route_option):
    apcdata = apcdata.where(apcdata.route_id == route_option)
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
        mycut = pd.cut(df['load'].tolist(), bins=bins)
        
    df['y_class'] = mycut.codes
    return df

def get_apc_data_for_date(filter_date):
    print("Running this...")
    filepath = os.path.join(os.getcwd(), "data", config.MTA_PARQUET)
    apcdata = spark.read.load(filepath)
    apcdata.createOrReplaceTempView("apc")
    plot_date = filter_date.strftime('%Y-%m-%d')    
    get_columns = ['trip_id', 'transit_date', 'departure_time', 'vehicle_id', 'ons',
                   'block_abbr', 'stop_sequence', 'stop_name', 'stop_id_original',
                   'load', 
                   'darksky_temperature', 
                   'darksky_humidity', 
                   'darksky_precipitation_probability', 
                   'route_direction_name', 'route_id', 'gtfs_direction_id',
                   'dayofweek',  'year', 'month', 'hour',
                   'sched_hdwy', 'map_latitude', 'map_longitude']
    get_str = ", ".join([c for c in get_columns])
    query = f"""
    SELECT {get_str}
    FROM apc
    WHERE (transit_date == '{plot_date}')
    ORDER BY departure_time
    """
    apcdata = spark.sql(query)
    apcdata = apcdata.withColumn("route_id_dir", F.concat_ws("_", apcdata.route_id, apcdata.route_direction_name))
    apcdata = apcdata.withColumn("day", F.dayofmonth(apcdata.departure_time))
    # apcdata = apcdata.drop("route_direction_name")
    apcdata = apcdata.withColumn("load", F.when(apcdata.load < 0, 0).otherwise(apcdata.load))
    return apcdata
    
st.title("String Plots")
st.sidebar.markdown("# Parameters")

with st.sidebar:
    sidebar_container = st.container()
    dataset_selectbox = st.selectbox('Dataset', ('Nashville, MTA', 'Chattanooga, CARTA'))
    if dataset_selectbox == 'Chattanooga, CARTA':
        fp = os.path.join('data', 'CARTA_route_ids.csv')
        filter_date = st.date_input('Filter dates', 
                                    min_value=dt.date(2019, 1, 1), max_value=dt.date(2022, 5, 30),
                                    value=dt.date(2021, 10, 18))
    elif dataset_selectbox == 'Nashville, MTA':
        filter_date = st.date_input('Filter dates', 
                                    min_value=dt.date(2020, 1, 1), max_value=dt.date(2022, 4, 6),
                                    value=dt.date(2021, 11, 1))
        apcdata = get_apc_data_for_date(filter_date)
        route_list = get_route_list(apcdata)
        route_option = st.selectbox('Route:', route_list)

    data_options = st.selectbox('Data to show:', ['Boardings', 'Occupancy'])
    if data_options == 'Occupancy':
        predict_time = st.time_input('Time to predict:', dt.time(12, 45))

container = st.container()
col1, col2 = container.columns(2)
with col1:
    st.subheader("First Y-Axis")
    enable1 = st.checkbox('Enable1?', True)
    if enable1:
        
        block_abbr_list = get_block_abbr_list(apcdata, route_option)
        block_options1 = st.multiselect('Block1:', block_abbr_list)
        direction_option_label1 = st.radio("Direction1:", ["To Downtown", "From Downtown"], horizontal=True)
        direction_option1 = 1 if direction_option_label1 == "To Downtown" else 0

with col2:
    st.subheader("Second Y-Axis")
    enable2 = st.checkbox('Enable2?', True)
    if enable2:
        block_abbr_list = get_block_abbr_list(apcdata, route_option)
        block_options2 = st.multiselect('Block2:', block_abbr_list)
        direction_option_label2 = st.radio("Direction2:", ["From Downtown", "To Downtown"], horizontal=True)
        direction_option2 = 1 if direction_option_label2 == "To Downtown" else 0
    
plot_button = st.button('Plot graphs')
    
if plot_button:
    with st.spinner("Processing..."):
        # First graph
        if enable1:
            # apcdata1 = apcdata.where(apcdata.block_abbr == int(block_option1))
            apcdata1 = apcdata.where(F.col("block_abbr").isin(block_options1))
            apcdata1 = apcdata1.where(apcdata1.gtfs_direction_id == direction_option1)
            if apcdata1.count() == 0:
                st.error("First plot is empty.")
            df1 = apcdata1.toPandas()
            df1['plot_no'] = 0
        
        # Second graph
        if enable2:
            # apcdata2 = apcdata.where(apcdata.block_abbr == int(block_option2))
            apcdata2 = apcdata.where(F.col("block_abbr").isin(block_options2))
            apcdata2 = apcdata2.where(apcdata2.gtfs_direction_id == direction_option2)
            if apcdata2.count() == 0:
                st.error("Second plot is empty.")
            df2 = apcdata2.toPandas()
            df2['plot_no'] = 1
        
        if enable1 and enable2:
            if (direction_option1 == direction_option2) and (block_options1 == block_options2):
                df = df1
            else:
                df = pd.concat([df1, df2])
            block_options = list(set(block_options1 + block_options2))
        elif enable1 and not enable2:
            df = df1
            block_options = block_options1
        elif not enable1 and enable2:
            df = df2
            block_options = block_options2
            
            
        df = assign_data_to_bins(df, data_options)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        df['geometry'] = df.apply(lambda x: Point(x['map_latitude'], x['map_longitude']), axis=1)
        df = gpd.GeoDataFrame(df, geometry='geometry')
        for direction in ['FROM DOWNTOWN', 'TO DOWNTOWN']:
            dir_df = df[df['route_direction_name'] == direction]
            trip_ids = dir_df.groupby(['trip_id']).first().sort_values(by=['departure_time']).reset_index().trip_id.tolist()
            r = lambda: random.randint(0,255)
            color = '#%02X%02X%02X' % (r(),r(),r())
            for trip_id in trip_ids:
                trip_df = dir_df[dir_df['trip_id'] == trip_id]
                marker_edges = [color,'#cccccc','#969696','#525252', '#f7f7f7','#cccccc','#969696','#525252', '#f7f7f7','#cccccc','#969696']
                trip_df['colors'] = trip_df['y_class'].apply(lambda x: marker_edges[x])
                # st.dataframe(trip_df[['departure_time', 'stop_sequence']])

                trip_df = trip_df.sort_values(by=['departure_time', 'stop_sequence'], ascending=direction != 'TO DOWNTOWN')
                trip_df['geom_ahead'] = trip_df['geometry'].shift(1)
                trip_df['distance_from_prev'] = trip_df['geometry'].distance(trip_df['geom_ahead'])
                trip_df['distance_cusum'] = trip_df['distance_from_prev'].cumsum()
                trip_df['distance_cusum'] = trip_df['distance_cusum'].fillna(0)
                trip_df = trip_df.dropna(subset=['departure_time']).reset_index(drop=True)
                fig.add_trace(go.Scatter(x=trip_df['departure_time'],
                                         y=trip_df['distance_cusum'],
                                         mode='lines+markers',
                                         showlegend=True,
                                         legendgroup=direction,
                                         legendgrouptitle_text=direction,
                                         name=trip_id,
                                         hoverlabel = dict(bgcolor=color),
                                         line=dict(color=color, width=3),
                                         marker=dict(line=dict(color='black', width=1.5), size=10, color=trip_df['colors']),
                                         customdata  = np.stack((trip_df['vehicle_id'], trip_df['stop_name'], trip_df['ons'], trip_df['route_direction_name']), axis=-1),
                                         hovertemplate = ('<i>Vehicle ID</i>: %{customdata[0]}'+\
                                                         '<br><b>Stop Name</b>: %{customdata[1]}'+\
                                                         '<br><b>Data</b>: %{customdata[2]}'+\
                                                         '<br><b>Direction</b>: %{customdata[3]}'+\
                                                         '<br><b>Time</b>: %{x|%H:%M:%S}<br><extra></extra>')),
                              secondary_y=direction == 'TO DOWNTOWN')
            if trip_ids:
                fig.update_yaxes(
                        title_text=f"<b>{direction}</b>",
                        tickmode = 'array',
                        tickvals = trip_df['distance_cusum'],
                        ticktext = trip_df['stop_id_original'],
                        secondary_y=direction == 'TO DOWNTOWN'
                )
                
        if enable1 and enable2:
            fig.update_layout(title=f'{data_options} for Block: {block_options1}, Route: {route_option} {direction_option_label1} and {block_options2} {direction_option_label2}')
        elif enable1 and not enable2:
            fig.update_layout(title=f'{data_options} for Block: {block_options1}, Route: {route_option} {direction_option_label1}')
        elif not enable1 and enable2:
            fig.update_layout(title=f'{data_options} for Block: {block_options2}, Route: {route_option} {direction_option_label2}')

        st.plotly_chart(fig)