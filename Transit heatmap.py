# Contents of ~/my_app/streamlit_app.py
import streamlit as st
import plotly.graph_objects as go
import datetime as dt
import numpy as np
import pandas as pd
from src import plot_utils
# import pickle5 as pickle

st.markdown("# Transit Heatmap")
st.sidebar.markdown("# Transit heatmap")

with st.sidebar:
    plot_demo_button = st.button('Plot example graph')
    time_granularity = st.number_input('Aggregation time (in minutes)', value=60)
    filter_date = st.date_input('Filter dates', value=(dt.date(2022, 1, 11), dt.date(2022, 2, 12)))
    plot_demo2_button = st.button('Plot graph from dataset')
    # months = [i for i in range(filter_date[0].month, 13)]
    # st.write([year for year in range(filter_date[0].year, filter_date[1].year+1)])
    
# st.write('The current number is ', aggregation_number)

def route_dir_func(row):
    return (row.route_id + row.row.route_direction_name)


if plot_demo2_button:
    # Load dataset
    years = [year for year in range(filter_date[0].year, filter_date[1].year+1)]
    apc_df = pd.DataFrame()
    if len(years)==1:
        months = [i for i in range(filter_date[0].month, (filter_date[1].month)+1)]
        # apc_df = pd.read_parquet('./data/carta_apc_out.parquet/year={}/month={}/'.format(years[0], months[0]), engine='pyarrow')
        for month in months:
            apc_df = apc_df.append(pd.read_parquet('./data/carta_apc_out.parquet/year={}/month={}/'.format(years[0], month), engine='pyarrow'))
    elif len(years)==2:
        months = [i for i in range(filter_date[0].month, 13)]
        # apc_df = pd.read_parquet('./data/carta_apc_out.parquet/year={}/month={}/'.format(years[0], months[0]))
        for month in months:
            apc_df = apc_df.append(pd.read_parquet('./data/carta_apc_out.parquet/year={}/month={}/'.format(years[0], month), engine='pyarrow'))

        months = [i for i in range(1, filter_date[1].month+1)]
        for month in months:
            apc_df = apc_df.append(pd.read_parquet('./data/carta_apc_out.parquet/year={}/month={}/'.format(years[1], month), engine='pyarrow'))
    
    elif len(years)>2:
        months = [i for i in range(filter_date[0].month, 13)]
        # apc_df = pd.read_parquet('./data/carta_apc_out.parquet/year={}/month={}/'.format(years[0], months[0]))
        for month in months:
            apc_df = apc_df.append(pd.read_parquet('./data/carta_apc_out.parquet/year={}/month={}/'.format(years[0], month), engine='pyarrow'))

        apc_df.append(pd.read_parquet('./data/carta_apc_out.parquet/year={}/'.format(years[1:-1]), engine='pyarrow'))

        months = [i for i in range(1, filter_date[1].month+1)]
        for month in months:
            apc_df = apc_df.append(pd.read_parquet('./data/carta_apc_out.parquet/year={}/month={}/'.format(years[1], month), engine='pyarrow'))

    # st.dataframe(apc_df.head())
    # st.dataframe(apc_df.tail())


    vis_df = apc_df

    vis_df.transit_date = pd.to_datetime(vis_df.transit_date)
    vis_df.time_actual_arrive = pd.to_datetime(vis_df.time_actual_arrive)
    vis_df = vis_df.dropna(subset=['time_actual_arrive'])

    if (vis_df.iloc[0].transit_date >= filter_date[1]) or (vis_df.iloc[-1].transit_date <= filter_date[0]):
        st.write('Date ranges out of data')
        exit()

    vis_df['transit_date'] = pd.to_datetime(vis_df.transit_date)
    

    start_date = pd.Timestamp(filter_date[0])
    end_date   = pd.Timestamp(filter_date[1])
    vis_df = vis_df[(vis_df['transit_date'] >= start_date) & 
                    (vis_df['transit_date'] <= end_date)]
    
    # plot dataframe data
    # st.dataframe(vis_df.head())
    
    fig = plot_utils.plot_max_aggregate(vis_df, start_date, end_date, time_granularity)
    st.plotly_chart(fig, use_container_width=True)


if plot_demo_button:
    # Load/Prepare dataframe data
    routes = 10
    start_date = pd.Timestamp(filter_date[0])
    end_date   = pd.Timestamp(filter_date[1])
    date_range = pd.date_range(start_date, end_date, freq='1D')
    data = np.random.randint(1, 5, size=(routes, len(date_range)))
    df = pd.DataFrame(data, columns=date_range)
    
    # This will print the dataframe on screen
    # st.dataframe(df)

    bvals = [0, 1, 2, 3, 4, 5]
    colors = ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026']
    dcolorsc = plot_utils.discrete_colorscale(bvals, colors)

    tickvals = [0, 1, 2, 3, 4, 5]
    ticktext = ['low', 'med', 'med-high', 'high', 'very-high']
        
        
    hovertext = list()
    for yi, yy in enumerate(df.index):
        hovertext.append(list())
        for xi, xx in enumerate(date_range):
            timestamp = xx.time()
            data = df.to_numpy()[yi][xi]
            if not np.isnan(data):
                load = ticktext[int(data)]
            else:
                load = data
            hovertext[-1].append('Time: {}<br />Route: {}<br />Load: {}'.format(timestamp, yy, load))
            
    fig = go.Figure(data=[go.Heatmap(z=df.to_numpy(), 
                                xgap=2,
                                ygap=2,
                                y=df.index,
                                x=date_range,
                                zmin=0, zmax=4, 
                                colorscale = dcolorsc, 
                                colorbar = dict(thickness=25, 
                                                tickvals=tickvals, 
                                                ticktext=ticktext),
                                showscale=True,
                                type = 'heatmap',
                                hoverongaps=False,
                                hoverinfo='text',
                                text=hovertext)])
    
    st.plotly_chart(fig, use_container_width=True)

    
