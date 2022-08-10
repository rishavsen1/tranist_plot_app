import streamlit as st
import datetime as dt
import os
from src import config
from src import stop_level_utils
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import joblib

marker_edges = ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026']
labels = ["0-5 pax", "6-11 pax", "12-16 pax", "17-29 pax", "30-100 pax"]
colors = ['#67001f','#b2182b','#d6604d','#f4a582','#fddbc7','#f7f7f7','#d1e5f0','#92c5de','#4393c3','#2166ac','#053061']

spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '40g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '20g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .getOrCreate()

# Data retrieval
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
                   'load', 'arrival_time', 
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

def add_features(df, TIMEWINDOW=15):
    df['day'] = df["arrival_time"].dt.day
    df = df.sort_values(by=['block_abbr', 'arrival_time'])#.reset_index(drop=True)

    # Adding extra features
    # Holidays
    fp = os.path.join(os.getcwd(), 'data', 'US Holiday Dates (2004-2021).csv')
    holidays_df = pd.read_csv(fp)
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])
    holidays_df['is_holiday'] = True
    df = df.merge(holidays_df[['Date', 'is_holiday']], left_on='transit_date', right_on='Date', how='left')
    df['is_holiday'] = df['is_holiday'].fillna(False)
    df = df.drop(columns=['Date'], axis=1)

    # Traffic
    # Causes 3M data points to be lost
    fp = os.path.join(os.getcwd(), 'data', 'triplevel_speed.pickle')
    speed_df = pd.read_pickle(fp)
    speed_df = speed_df.rename({'route_id_direction':'route_id_dir'}, axis=1)
    speed_df = speed_df[['transit_date', 'trip_id', 'route_id_dir', 'traffic_speed']]
    df = df.merge(speed_df, how='left', 
                  left_on=['transit_date', 'trip_id', 'route_id_dir'], 
                  right_on=['transit_date', 'trip_id', 'route_id_dir'])
    df['traffic_speed'].bfill(inplace=True)

    df = df.dropna(subset=['arrival_time'])
    df['minute'] = df['arrival_time'].dt.minute
    df['minuteByWindow'] = df['minute'] // TIMEWINDOW
    df['temp'] = df['minuteByWindow'] + (df['hour'] * 60 / TIMEWINDOW)
    df['time_window'] = np.floor(df['temp']).astype('int')
    df = df.drop(columns=['minute', 'minuteByWindow', 'temp'], axis=1)

    # HACK
    df = df[df['hour'] != 3]
    df = df[df['stop_sequence'] != 0]

    df = df.sort_values(by=['block_abbr', 'arrival_time'])#.reset_index(drop=True)
    return df

# Plotting
def calculate_distances(trip_df, direction):
    trip_df['colors'] = trip_df['y_class'].apply(lambda x: marker_edges[x])
    trip_df['line_color'] = trip_df['vehicle_idx'].apply(lambda x: colors[x])
    trip_df = trip_df.sort_values(by=['departure_time', 'stop_sequence'], ascending=direction != 'TO DOWNTOWN')
    trip_df['geom_ahead'] = trip_df['geometry'].shift(1)
    trip_df['distance_from_prev'] = trip_df['geometry'].distance(trip_df['geom_ahead'])
    trip_df['distance_cusum'] = trip_df['distance_from_prev'].cumsum()
    trip_df['distance_cusum'] = trip_df['distance_cusum'].fillna(0)
    trip_df = trip_df.dropna(subset=['departure_time']).reset_index(drop=True)
    trip_df = trip_df.drop(columns=['geometry', 'geom_ahead'], axis=1)
    return trip_df

def plot_boardings(df, directions):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    df_high = []
    # Plot lines and markers
    for i, direction in enumerate(directions):
        dir_df = df[df['route_direction_name'] == direction]
        trip_ids = dir_df.groupby(['trip_id']).first().sort_values(by=['departure_time']).reset_index().trip_id.tolist()
        for trip_id in trip_ids:
            trip_df = dir_df[dir_df['trip_id'] == trip_id]
            trip_df = calculate_distances(trip_df, direction)
            df_high.append(trip_df)
            fig = plot_markers_on_fig(trip_df, fig, direction, symbol='circle', data_column='ons')
        if trip_ids:
            fig.update_yaxes(
                    title_text=f"<b>{direction}</b>",
                    tickmode = 'array',
                    tickvals = trip_df['distance_cusum'],
                    ticktext = trip_df['stop_id_original'],
                    secondary_y=direction == 'TO DOWNTOWN'
            )

    df_high = pd.concat(df_high)
    color = trip_df.iloc[0]['line_color']
    plot_marker_data_legend(fig, df_high, color)

    return fig

def plot_occupancy(df, filter_date, predict_time, past=5, future=10):
    columns = joblib.load('data/mta_stop_level/LL_X_columns.joblib')
    label_encoders = joblib.load('data/mta_stop_level/LL_Label_encoders.joblib')
    ohe_encoder = joblib.load('data/mta_stop_level/LL_OHE_encoder.joblib')
    num_scaler = joblib.load('data/mta_stop_level/LL_Num_scaler.joblib')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    datetime_now = dt.datetime.combine(filter_date, predict_time)
    df = add_features(df, TIMEWINDOW=15)

    found_past = False
    found_future = False

    df_high = []
    for vehicle, vehicle_df in df.groupby('vehicle_id'):
        # st.write(f"In: {vehicle}")
        trip_ids = vehicle_df.groupby(['trip_id']).first().sort_values(by=['departure_time']).reset_index().trip_id.tolist()
        # Get latest trip only
        
        last_valid_past = pd.DataFrame()
        last_valid_future = pd.DataFrame()
        
        for trip_id in trip_ids:
            # st.write(f"trip_id: {trip_id}")
            trip_df = vehicle_df[vehicle_df['trip_id'] == trip_id]
            past_df = trip_df[trip_df['departure_time'] < datetime_now]
            future_df = trip_df[trip_df['departure_time'] >= datetime_now]
            
            trip_df['status'] = 'past'
            trip_df['prediction'] = False
            # TODO: Need to display the line between past and future
            trip_df.loc[future_df.index, 'status'] = 'future'
            
            fig, _trip_df = plot_trip_prediction_markers(trip_df, fig, status='past')
            df_high.append(_trip_df)
                    
            direction = trip_df.iloc[0]['route_direction_name']
            _trip_df = calculate_distances(trip_df, direction)
            fig.update_yaxes(
                            title_text=f"<b>{direction}</b>",
                            tickmode = 'array',
                            tickvals = _trip_df['distance_cusum'],
                            ticktext = _trip_df['stop_id_original'],
                            secondary_y=direction == 'TO DOWNTOWN'
                            )
        
            if len(past_df[-past:]) == past:
                # st.write(f"trip_id: {trip_id}, past: {past_df.shape}")
                last_valid_past = past_df
                found_past = True
            if len(future_df[0:future]) == future:
                # st.write(f"trip_id: {trip_id}, future: {future_df.shape}")
                found_future = True
                last_valid_future = future_df
                break
        
        if found_past and found_future:
            input_df = pd.concat([last_valid_past[-past:], last_valid_future[0:future]])
            input_df = prepare_input_data(input_df, ohe_encoder, label_encoders, num_scaler, columns, target='y_class')
            
            num_features = input_df.shape[1]
            model = stop_level_utils.get_model(num_features, num_classes=5, path='data/mta_stop_level/model')
            y_pred = stop_level_utils.generate_simple_lstm_predictions(input_df, model, past, future)
            trip_df.loc[last_valid_future[0:FUTURE].index, 'y_class'] = y_pred
            trip_df.loc[last_valid_future[0:FUTURE].index, 'prediction'] = True

        for dir, dir_df in trip_df.groupby('route_direction_name'):
            # st.write(f"Prediction for {vehicle}")
            # st.write(dir_df.shape)
            # st.dataframe(dir_df.drop(['geometry'], axis=1))
            dir_df['route_direction_name'] = dir
            fig, dir_df = plot_trip_prediction_markers(dir_df, fig=fig, data_column='y_class', symbol='hexagram', status='future')
            df_high.append(dir_df)
        
    df_high = pd.concat(df_high)
    color = trip_df.iloc[0]['line_color']
    fig = plot_marker_data_legend(fig, df_high, color)
    return fig

def plot_markers_on_fig(trip_df, fig, direction, symbol='circle', data_column='ons'):
    color = trip_df.iloc[0]['line_color']
    trip_id = trip_df.iloc[0]['trip_id']
    block_name = str(trip_df['block_abbr'].iloc[0])
    fig.add_trace(go.Scatter(x=trip_df['departure_time'],
                                y=trip_df['distance_cusum'],
                                mode='lines+markers',
                                showlegend=True,
                                legendgroup=block_name,
                                legendgrouptitle_text=f"{block_name} trips",
                                name=trip_id,
                                hoverinfo=None,
                                line=dict(color=color, width=3),
                                hoverlabel=dict(bgcolor=color),
                                marker=dict(line=dict(color='black', width=1.5), 
                                            size=10, symbol=symbol,
                                            color=trip_df['colors']),
                                customdata=np.stack((trip_df['block_abbr'], 
                                                    trip_df['vehicle_id'], 
                                                    trip_df['stop_name'], 
                                                    trip_df[data_column], 
                                                    trip_df['route_direction_name']), axis=-1),
                                hovertemplate = ('<b>Block</b>: %{customdata[0]}'+\
                                                '<br><b>Vehicle ID</b>: %{customdata[1]}'+\
                                                '<br><b>Stop Name</b>: %{customdata[2]}'+\
                                                '<br><b>Data</b>: %{customdata[3]}'+\
                                                '<br><b>Direction</b>: %{customdata[4]}'+\
                                                '<br><b>Departure Time</b>: %{x|%H:%M:%S}<br><extra></extra>')),
                    secondary_y=direction == 'TO DOWNTOWN')
    return fig

def plot_trip_prediction_markers(trip_df, fig, data_column='load', symbol='circle', status='past'):
    direction = trip_df.iloc[0]['route_direction_name']
    trip_df = calculate_distances(trip_df, direction)
    trip_df = trip_df[trip_df['status'] == status]
    if status == 'future':
        trip_df = trip_df[trip_df['prediction'] == True]
    if trip_df.empty:
        return fig, trip_df
    fig = plot_markers_on_fig(trip_df, fig, direction, symbol=symbol, data_column=data_column)
    return fig, trip_df

def plot_marker_data_legend(fig, df_high, color):
    for y_class, y_class_df in df_high.groupby("y_class"):
        y_class_df = y_class_df[0:1]
        direction = y_class_df['route_direction_name'].unique().tolist()[0]
        fig.add_trace(go.Scatter(x=y_class_df['departure_time'],
                                    y=y_class_df['distance_cusum'],
                                    mode='markers',
                                    showlegend=True,
                                    legendgroup=data_options,
                                    legendgrouptitle_text=data_options,
                                    name=labels[y_class],
                                    hoverlabel=dict(bgcolor=color),
                                    marker=dict(line=dict(color='black', width=1.5),
                                                size=10, 
                                                color=y_class_df['colors']),
                                    customdata=np.stack((y_class_df['block_abbr'], 
                                                        y_class_df['vehicle_id'], 
                                                        y_class_df['stop_name'], 
                                                        y_class_df['y_class'], 
                                                        y_class_df['route_direction_name']), axis=-1),
                                    hovertemplate = ('<b>Block</b>: %{customdata[0]}'+\
                                                    '<br><b>Vehicle ID</b>: %{customdata[1]}'+\
                                                    '<br><b>Stop Name</b>: %{customdata[2]}'+\
                                                    '<br><b>Data</b>: %{customdata[3]}'+\
                                                    '<br><b>Direction</b>: %{customdata[4]}'+\
                                                    '<br><b>Departure Time</b>: %{x|%H:%M:%S}<br><extra></extra>')),
                        secondary_y=direction == 'TO DOWNTOWN')
    return fig

# Prediction
def prepare_input_data(input_df, ohe_encoder, label_encoders, num_scaler, columns, keep_columns=[], target='y_class'):
    num_columns = ['darksky_temperature', 'darksky_humidity', 'darksky_precipitation_probability', 'sched_hdwy', 'traffic_speed']
    cat_columns = ['month', 'hour', 'day', 'stop_sequence', 'stop_id_original', 'year', 'time_window']
    ohe_columns = ['dayofweek', 'route_id_dir', 'is_holiday']

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

# STREAMLIT
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
        df['geometry'] = df.apply(lambda x: Point(x['map_latitude'], x['map_longitude']), axis=1)
        df = gpd.GeoDataFrame(df, geometry='geometry')
        
        # Assign indices to vehicles and blocks
        vehicle_list = df['vehicle_id'].unique().tolist()
        df['vehicle_idx'] = df['vehicle_id'].apply(lambda x: vehicle_list.index(x))
        df['block_idx'] = df['block_abbr'].apply(lambda x: block_options.index(x))
        
        directions = list(set([direction_option_label1.upper(), direction_option_label2.upper()]))
        
        if data_options == 'Boardings':
            fig = plot_boardings(df, directions)
        elif data_options == 'Occupancy':
            PAST = 5
            FUTURE = 10
            fig = plot_occupancy(df, filter_date, predict_time, past=PAST, future=FUTURE)
            
        if enable1 and enable2:
            fig.update_layout(title=f'{data_options} for Block: {block_options1}, Route: {route_option} {direction_option_label1} and {block_options2} {direction_option_label2}')
        elif enable1 and not enable2:
            fig.update_layout(title=f'{data_options} for Block: {block_options1}, Route: {route_option} {direction_option_label1}')
        elif not enable1 and enable2:
            fig.update_layout(title=f'{data_options} for Block: {block_options2}, Route: {route_option} {direction_option_label2}')
        
        fig.update_layout(legend=dict(groupclick="toggleitem"), xaxis_title="<b>Departure Times</b>",)
        st.plotly_chart(fig)