from time import strftime
import streamlit as st
import plotly.express as px
import datetime as dt
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

def func2(x):
    for i in range(len(time_arr)):
        try:
            if x<=time_arr[i]:
                return int(i)
        except:pass
    return None


st.markdown("# Route-wise Statistics")
st.sidebar.markdown("# Per Route-wise Statistics")

# Everytime you change something here, the entire site will refresh.
with st.sidebar:
    route_option = st.selectbox('Route:', ['1', '2A', '3', '4', '5A', '7', '8', '9', '10A', '10C', '10G', '13', '14', '15A', '16', '21', '28', '33', '34', 'DTS'])
    agg_time = st.number_input('Aggregation window (mins):', value=15)
    filter_date = st.date_input('Filter dates', value=(dt.date(2022, 1, 6), dt.date(2022, 2, 15)))
    # percentile_option = st.selectbox('Percentile:', (75, 90, 100))
    plot_button = st.button('Plot graphs')
    
if plot_button:
    time_arr = [dt.time() for dt in 
       datetime_range(datetime(2016, 9, 1, 0), datetime(2016, 9, 2, 1), 
       timedelta(minutes=agg_time))][1:-3]

    years = [year for year in range(filter_date[0].year, filter_date[1].year+1)]
    apc_df = pd.DataFrame()
    if len(years)==1:
        months = [i for i in range(filter_date[0].month, (filter_date[1].month)+1)]
        for month in months:
            apc_df = apc_df.append(pd.read_parquet('./data/carta_apc_out.parquet/year={}/month={}/'.format(years[0], month), engine='pyarrow'))
    elif len(years)==2:
        months = [i for i in range(filter_date[0].month, 13)]
        for month in months:
            apc_df = apc_df.append(pd.read_parquet('./data/carta_apc_out.parquet/year={}/month={}/'.format(years[0], month), engine='pyarrow'))

        months = [i for i in range(1, filter_date[1].month+1)]
        for month in months:
            apc_df = apc_df.append(pd.read_parquet('./data/carta_apc_out.parquet/year={}/month={}/'.format(years[1], month), engine='pyarrow'))
    
    elif len(years)>2:
        months = [i for i in range(filter_date[0].month, 13)]
        for month in months:
            apc_df = apc_df.append(pd.read_parquet('./data/carta_apc_out.parquet/year={}/month={}/'.format(years[0], month), engine='pyarrow'))

        apc_df.append(pd.read_parquet('./data/carta_apc_out.parquet/year={}/'.format(years[1:-1]), engine='pyarrow'))

        months = [i for i in range(1, filter_date[1].month+1)]
        for month in months:
            apc_df = apc_df.append(pd.read_parquet('./data/carta_apc_out.parquet/year={}/month={}/'.format(years[1], month), engine='pyarrow'))

    apc_df = apc_df.reset_index(drop=True)
    apc_df.transit_date = pd.to_datetime(apc_df.transit_date)
    
    if (apc_df.iloc[0].transit_date >= filter_date[1]) or (apc_df.iloc[-1].transit_date <= filter_date[0]):
        st.write('Date ranges out of data')
        exit()
    apc_df = apc_df[(apc_df['transit_date']>=pd.Timestamp(filter_date[0])) & (apc_df['transit_date']<=pd.Timestamp(filter_date[1]))]
    apc_df.time_actual_arrive = apc_df.time_actual_arrive.astype(str)
    apc_df.time_actual_arrive = pd.to_datetime(apc_df['time_actual_arrive'])

    df_route = apc_df[apc_df.route_id == route_option].reset_index(drop=True)

    df_route['time_only'] = pd.to_datetime(df_route['time_actual_arrive'],format= '%H:%M:%S' ).dt.time

    df_route['time_grp'] = df_route.time_only.apply(lambda x: func2(x))

    df_route.rename(columns={'route_direction_name':'direction'}, inplace=True)
    df_route.rename(columns={'ons':'boardings'}, inplace=True)

    # Showing the dataframe for easy reference for you
    df_route['time_axis'] = df_route.time_grp.dropna().apply(lambda x: (time_arr[int(x)]) ) 

    time_vals = df_route.sort_values('time_grp').time_grp.dropna().unique()

    int_arr=[]
    for i in range(len(time_vals)):
        if i%6 == 0:
            int_arr.append(time_vals[i])

    # st.dataframe(df_route.sort_values('actual_hdwy', ascending=False))

    st.write(f"max occupancy in {agg_time} minute windows")
    df_b = df_route.dropna(subset=['load']).groupby(['time_grp', 'trip_id', 'direction']).max().reset_index()
    fig = px.box(df_b, x="time_grp", y="load", facet_row="direction", color='direction', width=1000)
    fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = int_arr,
        ticktext = [time_arr[int(i)] for i in int_arr]
        )
    )
    st.plotly_chart(fig)

    st.write(f"boarding events in {agg_time} minute windows (scatter plot)")
    fig1 = px.scatter(df_route.sort_values('time_grp'), x="time_grp", y="boardings", facet_row="direction", color='direction', width=1000)
    fig1.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = int_arr,
        ticktext = [time_arr[int(i)] for i in int_arr]
        )
    )
    st.plotly_chart(fig1)

    st.write(f"headway (average gap between trips) in {agg_time} minute windows")
    fig1 = px.box(df_route.rename({'actual_hdwy':'headway'}, axis=1).sort_values('time_grp'), x="time_grp", y="headway", facet_row="direction", color='direction', width=1000, 
    labels={
                     "time_grp": "Aggregation interval (hh:mm:ss)",
                     "headway": "Headway (in seconds)",
                 }
                 )
    fig1.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = int_arr,
        ticktext = [time_arr[int(i)] for i in int_arr]
        )
    )
    st.plotly_chart(fig1)

    st.write(f"delays (scheduled vs actual time at the arrival at a stop) in {agg_time} minute windows")
    fig1 = px.box(df_route[(df_route.delay > -300) & (df_route.delay < 80000)].sort_values('time_grp'), x="time_grp", y="delay", facet_row="direction", color='direction', width=1000, 
    labels={
                     "time_grp": "Aggregation interval (hh:mm:ss)",
                     "delay": "Delay (in seconds)",
                 }
                 )
    fig1.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = int_arr,
        ticktext = [time_arr[int(i)] for i in int_arr]
        )
    )
    st.plotly_chart(fig1)

    st.write(f"boarding per stop per day in {agg_time} minute windows")
    df_stop = df_route.groupby(['stop_name', 'time_grp']).sum().reset_index()[['time_grp', 'stop_name', 'boardings']].dropna()
    fig1 = px.box(df_stop, x="stop_name", y="boardings", width=1000)
    fig1.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = int_arr,
        ticktext = [time_arr[int(i)] for i in int_arr]
        )
    )
    st.plotly_chart(fig1, use_container_width=False)