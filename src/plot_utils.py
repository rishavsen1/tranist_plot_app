import pandas as pd
from src import data_utils, stop_level_utils
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import datetime as dt
from copy import deepcopy

bvals = [0, 1, 2, 3, 4, 5]
colors = ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026']
tickvals = [0, 1, 2, 3, 4, 5]
ticktext = ['low:0-9', 'med:10-16', 'med-high:16-55', 'high:56-75', 'very-high:76-100']
# [(0.0, 9.0), (10.0, 16.0), (16.0, 55.0), (56.0, 75.0), (76.0, 100.0)]

line_colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99']
marker_edges = ['#f7f7f7','#cccccc','#969696','#525252', '#f7f7f7','#cccccc','#969696','#525252', '#f7f7f7','#cccccc','#969696']
color_scale = ['rgba(254,237,222,1.0)', 
               'rgba(253,190,133,1.0)',
               'rgba(253,141,60,1.0)',
               'rgba(217,71,1,1.0)']
MARKER_SIZE = 12
boardings_legend = {0:'0-6 pax', 1:'7-10 pax', 2:'11-15 pax', 3:'16-100 pax'}

def discrete_colorscale(bvals, colors):
    """
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """
    if len(bvals) != len(colors)+1:
        raise ValueError('len(boundary values) should be equal to  len(colors)+1')
    bvals = sorted(bvals)     
    nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]  #normalized values
    
    dcolorscale = [] #discrete colorscale
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])
    return dcolorscale

def plot_max_aggregate(vis_df, DATE_START, DATE_END, time_granularity, dataset_selectbox):
    all_route_id_dirs = vis_df.route_id_dir.unique()
    
    TIME_GRANULARITY = time_granularity #minutes
    time_range = pd.date_range(pd.Timestamp(DATE_START.strftime('%Y-%m-%d') + ' 00:00:00'),
                               pd.Timestamp(DATE_START.strftime('%Y-%m-%d') + ' 23:59:59'),
                               freq=f"{TIME_GRANULARITY}min")
    zero_arr = np.full((len(all_route_id_dirs), len(time_range)), np.nan)
    trip_id_arr = np.full((len(all_route_id_dirs), len(time_range)), np.nan)

    for i, route_id_dir in enumerate(all_route_id_dirs):
        _df = vis_df[vis_df['route_id_dir'] == route_id_dir]
        if _df.empty:
            continue
    
        _df['time_window'] = _df.apply(lambda x: data_utils.get_time_window(x, TIME_GRANULARITY, row_name='arrival_time'), axis=1)
        _df = _df[['time_window', 'load', 'trip_id']].groupby('time_window').agg({'load':'max', 'trip_id':'first'})
        indices = _df.index.astype('int')
        _df['y_classes'] = _df['load'].apply(lambda x: data_utils.get_class(x, dataset_selectbox))
            
        y_classes = _df['y_classes'].to_numpy()
        zero_arr[i, indices] = y_classes
        trip_id_arr[i, indices] = _df.trip_id.to_numpy(dtype=int)
        
    res_df = pd.DataFrame(zero_arr, columns=time_range)

    res_df.index = all_route_id_dirs
    # Drop rows with all nan
    res_df = res_df.dropna(how='all')
    # Drop cols with all nan
    res_df = res_df.dropna(axis=1, how='all')

    trip_id_df = pd.DataFrame(trip_id_arr, columns=time_range)
    trip_id_df.index = all_route_id_dirs
    trip_id_df = trip_id_df.dropna(how='all')
    trip_id_df = trip_id_df.dropna(axis=1, how='all')
    trip_id_df = trip_id_df.sort_index()

    res_df = res_df.sort_index()
    dcolorsc = discrete_colorscale(bvals, colors)
    hovertext = list()
    for yi, yy in enumerate(res_df.index):
        hovertext.append(list())
        for xi, xx in enumerate(res_df.columns):
                timestamp = xx.time()
                data = res_df.to_numpy()[yi][xi]
                trip_id = trip_id_df.to_numpy()[yi][xi]
                if not np.isnan(data):
                    load = ticktext[int(data)]
                    trip_id = int(trip_id_df.to_numpy()[yi][xi])
                else:
                    load = data
                    trip_id = trip_id
                if TIME_GRANULARITY == 1:
                    hovertext[-1].append('Trip ID: {}<br />Time: {}<br />Route: {}<br />Load: {}'.format(trip_id, timestamp, yy, load))
                else:
                    hovertext[-1].append('Time: {}<br />Route: {}<br />Load: {}'.format(timestamp, yy, load))
                
    fig = go.Figure(data=[go.Heatmap(z=res_df.to_numpy(),
                            y=res_df.index,
                            x=res_df.columns,
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
    fig.update_layout(
        xaxis_title="Max of aggregated 100th pct trip loads in the time window",
        yaxis_title="Active route id and directions",
        legend_title="Legend Title",
        xaxis_tickformat='%H:%M',
        xaxis_hoverformat='%H:%M',
    )
    return fig

def plot_prediction_heatmap(predict_date, prediction_df):
    dates = pd.date_range(pd.Timestamp(predict_date), pd.Timestamp(predict_date) + pd.Timedelta('24h') - pd.Timedelta('30m'), freq='30min')
    
    hovertext = list()
    for yi, yy in enumerate(prediction_df.index):
        hovertext.append(list())
        for xi, xx in enumerate(dates):
            timestamp = xx.time()
            data = prediction_df.to_numpy()[yi][xi]
            if not np.isnan(data):
                load = ticktext[int(data)]
            else:
                load = data
            hovertext[-1].append('Time: {}<br />Route: {}<br />Load: {}'.format(timestamp, yy, load))

    dcolorsc = discrete_colorscale(bvals, colors)
    fig = go.Figure(data=[go.Heatmap(z=prediction_df.to_numpy(), 
                                xgap=0.5,
                                ygap=0.5,
                                y=prediction_df.index,
                                x=dates,
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
    fig.update_layout(
        title=f"Prediction for {predict_date} with 30 minute time windows.",
        xaxis_title="Time",
        yaxis_title="Route id and directions",
        xaxis_tickformat='%H:%M',
        xaxis_hoverformat='%H:%M'
    )
    return fig

# TODO: Might move to data utils
def prepare_vehicle_route_data(df, vehicle):
    tdf = df[(df['vehicle_id'] == vehicle)]
    # Fixing null arrival times (for completeness of axes labels)
    tdf['valid'] = 1
    tdf.loc[tdf['arrival_time'].isnull(), 'valid'] = 0

    end_stop = tdf.stop_sequence.max()
    tdf = tdf[tdf.stop_sequence != end_stop].reset_index(drop=True)

    tdf['orig_ss'] = tdf['stop_sequence']
    tdf.loc[tdf['gtfs_direction_id'] == 1, 'stop_sequence'] = abs(tdf.loc[tdf['gtfs_direction_id'] == 1, 'stop_sequence'] - tdf.loc[tdf['gtfs_direction_id'] == 1, 'stop_sequence'].max() - 1)

    tdf = tdf.sort_values(by=['trip_id', 'stop_sequence'])
    tdf['arrival_time'] = pd.to_datetime(tdf['arrival_time'].interpolate('bfill'))
    tdf = tdf.sort_values(by=['arrival_time', 'stop_sequence'])

    # TODO: Hard coded magic number
    if len(tdf) < 30:
        st.error(f"Vehicle {vehicle} has an empty dataset.")
        return pd.DataFrame()

    # Fixing issues where last and first stops gets interchanged
    init_stops = []
    for _, t_id_df in tdf.groupby('trip_id'):
        first_stop = t_id_df.sort_values('arrival_time').iloc[0].arrival_time
        init_stops.append({'key':first_stop, 'df':t_id_df.sort_values(by='orig_ss')})
    init_stops.sort(key=lambda x:x['key'])
    tdf_arr = [d['df'] for d in init_stops]
    tdf = pd.concat(tdf_arr)
    return tdf

def plot_string_boarding(df, vehicle_options):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    for v_idx, vehicle in enumerate(vehicle_options):
        tdf = prepare_vehicle_route_data(df, vehicle)
        st.write(v_idx, vehicle)
        st.dataframe(tdf)
        if tdf.empty:
            continue
        
        plot_mta_line_over_markers(fig, tdf, v_idx, vehicle)
        zero_bins_tdf = tdf[tdf['y_class'] == 0]
        plot_mta_markers_on_fig(fig, zero_bins_tdf, 'circle', v_idx, name=f"VId:{vehicle}")
        for bin, _df in tdf.groupby('y_class'):
            if bin > 0:
                plot_mta_markers_on_fig(fig, _df, 'hexagram', bin, vehicle, colorscale=color_scale, name=boardings_legend[bin])
        setup_fig_legend(fig, tdf)
    return fig
    
def plot_string_occupancy(df, plot_date, vehicle_options, predict_time=None):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    for v_idx, vehicle in enumerate(vehicle_options):
        tdf = prepare_vehicle_route_data(df, vehicle)
        if tdf.empty:
            continue
        future_tdf = pd.DataFrame()
        # TODO: Don't plot "future" markers
        if predict_time:
            time_now = predict_time
        else:
            time_now = dt.time(10, 37)
        datetime_now = dt.datetime.combine(plot_date, time_now)
        if tdf.iloc[-1]['arrival_time'] > datetime_now:
            past_df, to_predict_df = stop_level_utils.setup_past_future_from_datetime(tdf, datetime_now)
            if not to_predict_df.empty:
                plot_mta_markers_on_fig(fig, to_predict_df, 'hexagram-open', v_idx, vehicle, size=MARKER_SIZE+1, name='prediction')
            if not past_df.empty:
                future_tdf = tdf[tdf['arrival_time'] > to_predict_df.iloc[-1]['arrival_time']]
                tdf = tdf[tdf['arrival_time'] <= past_df.iloc[-1]['arrival_time']]
                
                plot_mta_line_over_markers(fig, tdf, v_idx, vehicle)
                plot_mta_line_over_markers(fig, future_tdf, v_idx, vehicle, dash='dash', width=1)
                
                plot_mta_markers_on_fig(fig, tdf, 'circle', v_idx, vehicle)
                plot_mta_markers_on_fig(fig, future_tdf, 'circle-open', v_idx, vehicle, name='no_info')
                setup_fig_legend(fig, tdf)
            else:
                future_tdf = deepcopy(tdf)
                tdf = pd.DataFrame()
                setup_fig_legend(fig, future_tdf)
            
    return fig

def setup_fig_legend(fig, tdf):
    tdf0 = tdf[tdf['gtfs_direction_id'] == 0].reset_index(drop=True)
    tdf1 = tdf[tdf['gtfs_direction_id'] == 1].reset_index(drop=True)
        
    if not tdf0.empty:
        tdf0_dict = dict(title="TO DOWNTOWN",
                         tickmode = 'array',
                         tickvals = tdf0.iloc[0:tdf0['stop_sequence'].max()].stop_sequence.tolist(),
                         ticktext = tdf0.iloc[0:tdf0['stop_sequence'].max()].stop_id_original.tolist())
    else:
        tdf0_dict = {}
    if not tdf1.empty:
        tdf1_dict = dict(title="FROM DOWNTOWN",
                         tickmode = 'array',
                         tickvals = tdf1.iloc[0:tdf1['stop_sequence'].max()].stop_sequence.tolist(),
                         ticktext = tdf1.iloc[0:tdf1['stop_sequence'].max()].stop_id_original.tolist())
    else:
        tdf1_dict = {}
        
    fig.update_layout(showlegend=True, 
                        # width=1500, height=1000,
                        yaxis = tdf0_dict,
                        yaxis2 = tdf1_dict,
                        legend={'traceorder':'normal'})
    
def plot_mta_line_over_markers(fig, df, v_idx, vehicle=None, dash='solid', width=4):
        # Plotting a line through the markers
    for t, t_id_df in df.groupby('trip_id'):
        if t_id_df['gtfs_direction_id'].any() == 1:
            secondary = True
        else:
            secondary = False
        valid_tdf = t_id_df[t_id_df['valid'] == 1]
        fig.add_trace(go.Scatter(x=valid_tdf['arrival_time'], y=valid_tdf['stop_sequence'],
                                line=dict(color=line_colors[v_idx], width=width, dash=dash),
                                mode='lines', 
                                showlegend = False,
                                hoverinfo='none', fillcolor=line_colors[v_idx]), secondary_y=secondary)
        
# TODO: Change direction and column names for uniformity with chattanooga
def plot_mta_markers_on_fig(fig, df, symbol, v_idx, vehicle=None, colorscale=colors, size=MARKER_SIZE, name='Vehicle'):
    ############################### TO DOWNTOWN ###############################
    tdf0 = df[df['gtfs_direction_id'] == 0].reset_index(drop=True)
    if tdf0.empty:
        showlegend1 = True
    showlegend1 = False
    tdf0['direction'] = "To Downtown"
    fig.add_trace(go.Scatter(x=tdf0['arrival_time'], y=tdf0['stop_sequence'],
                        mode='markers',
                        name=f"{name}",
                        yaxis="y1",opacity=1.0, fillcolor='rgba(0, 0, 0, 1.0)',
                        marker_size=tdf0["valid"]*size, marker_symbol=symbol,
                        marker=dict(line=dict(color='black', width=1),opacity=1.0,
                                    color=colorscale[v_idx]),
                        customdata  = np.stack((tdf0['vehicle_id'], tdf0['stop_name'], tdf0['ons'], tdf0['direction']), axis=-1),
                        hovertemplate = ('<i>Vehicle ID</i>: %{customdata[0]}'+\
                                        '<br><b>Stop Name</b>: %{customdata[1]}'+\
                                        '<br><b>Boardings</b>: %{customdata[2]}'+\
                                        '<br><b>Direction</b>: →%{customdata[3]}'+\
                                        '<br><b>Time</b>: %{x|%H:%M:%S}<br><extra></extra>')))
    ############################### FROM DOWNTOWN ###############################
    tdf1 = df[df['gtfs_direction_id'] == 1].reset_index(drop=True)
    tdf1['direction'] = "From Downtown"
    fig.add_trace(go.Scatter(x=tdf1['arrival_time'], y=tdf1['stop_sequence'],
                        mode='markers',
                        name=f"{name}",
                        showlegend = showlegend1,
                        yaxis="y2",opacity=1.0, fillcolor='rgba(0, 0, 0, 1.0)',
                        marker_size=tdf1["valid"]*size, marker_symbol=symbol,
                        marker=dict(line=dict(color='black', width=1),opacity=1.0,
                                    color=colorscale[v_idx]),
                        customdata  = np.stack((tdf1['vehicle_id'], tdf1['stop_name'], tdf1['ons'], tdf1['direction']), axis=-1),
                        hovertemplate = ('<i>Vehicle ID</i>: %{customdata[0]}'+\
                                        '<br><b>Stop Name</b>: %{customdata[1]}'+\
                                        '<br><b>Boardings</b>: %{customdata[2]}'+\
                                        '<br><b>Direction</b>: ←%{customdata[3]}'+\
                                        '<br><b>Time</b>: %{x|%H:%M:%S}<br><extra></extra>')), 
            secondary_y=True)