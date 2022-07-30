import streamlit as st
import datetime as dt
import os
from src import config
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '40g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '20g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .getOrCreate()

colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99']
marker_edges = ['#f7f7f7','#cccccc','#969696','#525252', '#f7f7f7','#cccccc','#969696','#525252', '#f7f7f7','#cccccc','#969696']
MARKER_SIZE = 12
color_scale = ['rgba(254,237,222,1.0)', 
               'rgba(253,190,133,1.0)',
               'rgba(253,141,60,1.0)',
               'rgba(217,71,1,1.0)']

def update_opacity(figure,opacity):
    for trace in range(len(figure['data'])):
        # print(figure['data'][trace]['fillcolor'],'-> ',end='')
        rgba_split = figure['data'][trace]['fillcolor'].split(',')
        figure['data'][trace]['fillcolor'] = ','.join(rgba_split[:-1] + [' {})'.format(opacity)])
        # print(figure['data'][trace]['fillcolor'])
    return figure

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
        
        # fp = os.path.join('data', 'MTA_route_ids.csv')
        # route_list = pd.read_csv(fp).dropna().sort_values('route_id').route_id.tolist()
        route_list = get_route_list(filter_date)
        route_option = st.selectbox('Route:', route_list)
        vehicle_list = get_vehicle_list(route_option)
        vehicle_options = st.multiselect('Vehicles:', vehicle_list)
        vehicle_options = [str(v) for v in vehicle_options]
        
        data_options = st.selectbox('Data to show:', ['Boardings', 'Occupancy'])
        print(vehicle_options)
    plot_button = st.button('Plot graphs')
    
if plot_button:
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
        st.dataframe(df)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        for v_idx, vehicle in enumerate(vehicle_options):
            tdf = df[(df['vehicle_id'] == vehicle)]

            tdf['color'] = 0
            tdf['valid'] = 1
            tdf.loc[tdf['arrival_time'].isnull(), 'valid'] = 0
            
            end_stop = tdf.stop_sequence.max()
            tdf = tdf[tdf.stop_sequence != end_stop].reset_index(drop=True)

            tdf['orig_ss'] = tdf['stop_sequence']
            tdf.loc[tdf['gtfs_direction_id'] == 1, 'stop_sequence'] = abs(tdf.loc[tdf['gtfs_direction_id'] == 1, 'stop_sequence'] - tdf.loc[tdf['gtfs_direction_id'] == 1, 'stop_sequence'].max() - 1)
            
            tdf = tdf.sort_values(by=['trip_id', 'stop_sequence'])
            tdf['arrival_time'] = pd.to_datetime(tdf['arrival_time'].interpolate('bfill'))
            tdf = tdf.sort_values(by=['arrival_time', 'stop_sequence'])
            
            if len(tdf) < 30:
                st.error(f"Vehicle {vehicle} has an empty dataset.")
                break
            
            init_stops = []
            for _, t_id_df in tdf.groupby('trip_id'):
                first_stop = t_id_df.sort_values('arrival_time').iloc[0].arrival_time
                init_stops.append({'key':first_stop, 'df':t_id_df.sort_values(by='orig_ss')})
            init_stops.sort(key=lambda x:x['key'])

            tdf_arr = [d['df'] for d in init_stops]
            tdf = pd.concat(tdf_arr)

            for t, t_id_df in tdf.groupby('trip_id'):
                if t_id_df['gtfs_direction_id'].any() == 1:
                    secondary = True
                else:
                    secondary = False
                valid_tdf = t_id_df[t_id_df['valid'] == 1]
                fig.add_trace(go.Scatter(x=valid_tdf['arrival_time'], y=valid_tdf['stop_sequence'],
                                        line=dict(color=colors[v_idx], width=4),
                                        mode='lines', 
                                        showlegend = False,
                                        hoverinfo='none', fillcolor=colors[v_idx]), secondary_y=secondary)

            ############################### FROM DOWNTOWN ###############################
            tdf0 = tdf[tdf['gtfs_direction_id'] == 0].reset_index(drop=True)
            fig.add_trace(go.Scatter(x=tdf0['arrival_time'], y=tdf0['stop_sequence'],
                                    mode='markers',
                                    name=f"Vehicle:{vehicle}",
                                    yaxis="y1",opacity=1.0, fillcolor='rgba(0, 0, 0, 1.0)',
                                    marker_size=tdf0["valid"]*MARKER_SIZE,
                                    marker=dict(line=dict(color='black', width=1),opacity=1.0,
                                                color=colors[v_idx]),
                                    customdata  = np.stack((tdf0['vehicle_id'], tdf0['stop_name'], tdf0['ons']), axis=-1),
                                    hovertemplate = ('<i>Vehicle ID</i>: %{customdata[0]}'+\
                                                    '<br><b>Stop Name</b>: %{customdata[1]}'+\
                                                    '<br><b>Boardings</b>: %{customdata[2]}'+\
                                                    '<br><b>Time</b>: %{x|%H:%M:%S.%L}<br>')))

            for bin, _df in tdf0.groupby('y_class'):
                if bin == 0:
                    continue
                fig.add_trace(go.Scatter(x=_df['arrival_time'], y=_df['stop_sequence'],
                                mode='markers',
                                showlegend = False,
                                marker_size=_df["valid"]*MARKER_SIZE, marker_symbol='hexagram',
                                marker=dict(line=dict(color='black', width=1),opacity=1.0,
                                color=color_scale[bin]),
                                customdata  = np.stack((_df['vehicle_id'], _df['stop_name'], _df['ons']), axis=-1),
                                hovertemplate = ('<i>Vehicle ID</i>: %{customdata[0]}'+\
                                                '<br><b>Stop Name</b>: %{customdata[1]}'+\
                                                '<br><b>Boardings</b>: %{customdata[2]}'+\
                                                '<br><b>Time</b>: %{x|%H:%M:%S.%L}<br>')), secondary_y=False)
            ############################### FROM DOWNTOWN ###############################
            tdf1 = tdf[tdf['gtfs_direction_id'] == 1].reset_index(drop=True)
            fig.add_trace(go.Scatter(x=tdf1['arrival_time'], y=tdf1['stop_sequence'],
                                    mode='markers',
                                    name=f"Vehicle:{vehicle}",
                                    showlegend = False,
                                    yaxis="y2",opacity=1.0, fillcolor='rgba(0, 0, 0, 1.0)',
                                    marker_size=tdf1["valid"]*MARKER_SIZE,
                                    marker=dict(line=dict(color='black', width=1),opacity=1.0,
                                                color=colors[v_idx]),
                                    customdata  = np.stack((tdf1['vehicle_id'], tdf1['stop_name'], tdf1['ons']), axis=-1),
                                    hovertemplate = ('<i>Vehicle ID</i>: %{customdata[0]}'+\
                                                    '<br><b>Stop Name</b>: %{customdata[1]}'+\
                                                    '<br><b>Boardings</b>: %{customdata[2]}'+\
                                                    '<br><b>Time</b>: %{x|%H:%M:%S.%L}<br>')), secondary_y=True)

            for bin, _df in tdf1.groupby('y_class'):
                if bin == 0:
                    continue
                fig.add_trace(go.Scatter(x=_df['arrival_time'], y=_df['stop_sequence'],
                                mode='markers',
                                # showlegend = False,
                                name=f"Bin:{bin}",
                                marker_size=_df["valid"]*MARKER_SIZE, marker_symbol='hexagram',
                                marker=dict(line=dict(color='black', width=1),opacity=1.0,
                                color=color_scale[bin]),
                                customdata  = np.stack((_df['vehicle_id'], _df['stop_name'], _df['ons']), axis=-1),
                                hovertemplate = ('<i>Vehicle ID</i>: %{customdata[0]}'+\
                                                '<br><b>Stop Name</b>: %{customdata[1]}'+\
                                                '<br><b>Boardings</b>: %{customdata[2]}'+\
                                                '<br><b>Time</b>: %{x|%H:%M:%S.%L}<br>')), secondary_y=True)
            #############################################################################################
            if not tdf0.empty:
                tdf0_dict = dict(
                                    title="TO DOWNTOWN",
                                    tickmode = 'array',
                                    tickvals = tdf0.iloc[0:tdf0['stop_sequence'].max()].stop_sequence.tolist(),
                                    ticktext = tdf0.iloc[0:tdf0['stop_sequence'].max()].stop_id_original.tolist()
                                )
            else:
                tdf0_dict = {}
            if not tdf1.empty:
                tdf1_dict = dict(
                                    title="FROM DOWNTOWN",
                                    tickmode = 'array',
                                    tickvals = tdf1.iloc[0:tdf1['stop_sequence'].max()].stop_sequence.tolist(),
                                    ticktext = tdf1.iloc[0:tdf1['stop_sequence'].max()].stop_id_original.tolist()
                                )
            fig.update_layout(showlegend=True, 
                                # width=1500, height=1000,
                                yaxis = tdf0_dict,
                                yaxis2 = tdf1_dict)


            
        st.plotly_chart(fig)