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
ticktext = ['low', 'med', 'med-high', 'high', 'very-high']

line_colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99']
marker_edges = ['#f7f7f7','#cccccc','#969696','#525252', '#f7f7f7','#cccccc','#969696','#525252', '#f7f7f7','#cccccc','#969696']
MARKER_SIZE = 12
boardings_legend = {0:'0-5 pax', 1:'5-11 pax', 2:'12-16 pax', 3:'17-29 pax', 4:'30-100 pax'}
# [(-1, 5), (5, 11), (11, 16), (16, 29), (29, 101)]
color_scale = ['rgba(122,122,122,255)','rgba(254,204,92,255)','rgba(253,141,60,255)','rgba(240,59,32,255)','rgba(189,0,38,255)']

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
def prepare_vehicle_route_data(df, vehicle_id):
    tdf = df[(df['vehicle_id'] == vehicle_id)]
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

    if len(tdf) == 0:
        st.error(f"Vehicle {vehicle_id} has an empty dataset.")
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

def plot_string_boarding(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    for v_idx, (vehicle_id, vehicle_df) in enumerate(df.groupby('vehicle_id')):
        tdf = prepare_vehicle_route_data(vehicle_df, vehicle_id)
        if tdf.empty:
            continue
        
        plot_mta_line_over_markers(fig, tdf, v_idx, vehicle_id, showlegend=True)
        for bin, _df in tdf.groupby('y_class'):
            opacity = 0.8
            if bin > 0:
                opacity = 1.0
            _df = _df[_df['valid'] == 1]
            plot_mta_markers_on_fig(fig, _df, 
                                    marker_symbol='circle', 
                                    marker_color=color_scale[bin], 
                                    marker_size=MARKER_SIZE, 
                                    legend_name=boardings_legend[bin],
                                    opacity=opacity,
                                    hover_bgcolor=line_colors[v_idx])
        setup_fig_legend(fig, tdf)
    return fig
    
def plot_string_occupancy(df, plot_date, predict_time=None):    
    past = 5
    future = 10
    
    occupancy_legend = {0:'0-6 pax', 1:'7-12 pax', 2:'13-100 pax'}
        
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    for v_idx, (vehicle_id, vehicle_df) in enumerate(df.groupby('vehicle_id')):
            tdf = prepare_vehicle_route_data(vehicle_df, vehicle_id)
            if tdf.empty:
                continue
            future_tdf = pd.DataFrame()
            # TODO: Don't plot "future" markers
            time_now = predict_time
            datetime_now = dt.datetime.combine(plot_date, time_now)
            if tdf.iloc[-1]['arrival_time'] > datetime_now:
                past_df, to_predict_df = stop_level_utils.setup_past_future_from_datetime(tdf, datetime_now, past=5, future=10)
                if not past_df.empty:
                    future_tdf = tdf[tdf['arrival_time'] > to_predict_df.iloc[-1]['arrival_time']]
                    tdf = tdf[tdf['arrival_time'] <= past_df.iloc[-1]['arrival_time']]
                    
                    plot_mta_line_over_markers(fig, tdf, v_idx, vehicle_id, showlegend=True)
                    plot_mta_line_over_markers(fig, future_tdf, v_idx, vehicle_id, dash='dash', width=1)
                    
                    for bin, _df in tdf.groupby('y_class'):
                        opacity = 0.8
                        if bin > 0:
                            opacity = 1.0
                        plot_mta_markers_on_fig(fig, _df, 
                                                marker_symbol='circle', 
                                                marker_color=color_scale[bin], 
                                                marker_size=MARKER_SIZE, 
                                                legend_name=occupancy_legend[bin],
                                                opacity=opacity,
                                                hover_bgcolor=color_scale[bin],
                                                data_column='load')
                    plot_mta_markers_on_fig(fig, future_tdf, 'circle-open', showlegend=False, 
                                            marker_color=line_colors[v_idx])
                    
                    setup_fig_legend(fig, tdf)
                else:
                    future_tdf = deepcopy(tdf)
                    tdf = pd.DataFrame()
                    setup_fig_legend(fig, future_tdf)
                if not to_predict_df.empty:
                    plot_mta_line_over_markers(fig, to_predict_df, v_idx, vehicle_id, dash='solid', width=3)
                    
                    ### PREDICTION ###
                    if not past_df.empty:
                        input_df = pd.concat([past_df, to_predict_df])
                        
                        keep_columns=['route_id_dir', 'trip_id']
                        input_df = stop_level_utils.prepare_input_data(input_df, keep_columns=keep_columns)
                        input_df = input_df.drop(columns=keep_columns)
                        num_features = input_df.shape[1]
                        model = stop_level_utils.get_model(num_features)

                        y_pred = stop_level_utils.generate_simple_lstm_predictions(input_df, model, past, future)
                        to_predict_df['y_class'] = y_pred
                        
                        for bin, _df in to_predict_df.groupby('y_class'):
                            opacity = 0.8
                            if bin > 0:
                                opacity = 1.0
                            plot_mta_markers_on_fig(fig, to_predict_df, 
                                                    marker_symbol='square', 
                                                    marker_color=color_scale[bin], 
                                                    marker_size=MARKER_SIZE, 
                                                    legend_name=occupancy_legend[bin],
                                                    hover_bgcolor=line_colors[v_idx],
                                                    data_column='y_class')
                    else:
                        st.error("Past dataframe is empty.")
            else:
                st.error("Too early to predict")
    return fig

def setup_fig_legend(fig, tdf):
    tdf0 = tdf[tdf['plot_no'] == 0].reset_index(drop=True)
    tdf1 = tdf[tdf['plot_no'] == 1].reset_index(drop=True)
    if not tdf0.empty:
        title = "TO DOWNTOWN" if tdf0.iloc[0]['gtfs_direction_id'] == 0 else "FROM DOWNTOWN"
        tdf0_dict = dict(title=title,
                         tickmode = 'array',
                         tickvals = tdf0.iloc[0:tdf0['stop_sequence'].max()].stop_sequence.tolist(),
                         ticktext = tdf0.iloc[0:tdf0['stop_sequence'].max()].stop_id_original.tolist())
    else:
        tdf0_dict = {}
    if not tdf1.empty:
        title = "TO DOWNTOWN" if tdf1.iloc[0]['gtfs_direction_id'] == 0 else "FROM DOWNTOWN"
        tdf1_dict = dict(title=title,
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
    
def plot_mta_line_over_markers(fig, df, v_idx, vehicle=None, dash='solid', width=4, showlegend=False):
        # Plotting a line through the markers
    for t, t_id_df in df.groupby('trip_id'):
        if t_id_df['plot_no'].any() == 1:
            secondary = True
        else:
            secondary = False
        valid_tdf = t_id_df[t_id_df['valid'] == 1]
        fig.add_trace(go.Scatter(x=valid_tdf['arrival_time'], y=valid_tdf['stop_sequence'],
                                 opacity=1.0,
                                line=dict(color=line_colors[v_idx], width=width, dash=dash),
                                mode='lines', 
                                showlegend=showlegend,
                                name=f"trip:{t}",
                                hoverinfo='none', fillcolor=line_colors[v_idx]), secondary_y=secondary)
        
# TODO: Change direction and column names for uniformity with chattanooga
# def plot_mta_markers_on_fig(fig, df, symbol, v_idx, vehicle=None, colorscale=line_colors, size=MARKER_SIZE, name='Vehicle', opacity=1.0):
def plot_mta_markers_on_fig(fig, df, marker_symbol, marker_color, marker_size=MARKER_SIZE, legend_name='Vehicle', opacity=1.0, hover_bgcolor=None, showlegend=True, data_column='ons'):
    ############################### TO DOWNTOWN ###############################
    tdf0 = df[df['plot_no'] == 0].reset_index(drop=True)
    tdf0['direction'] = "→To Downtown" if (tdf0['gtfs_direction_id'] == 0).any() else "←From Downtown"
    fig.add_trace(go.Scatter(x=tdf0['arrival_time'], y=tdf0['stop_sequence'],
                             mode='markers',
                             name=f"left {legend_name}",
                             yaxis="y1", opacity=opacity, fillcolor='rgba(0, 0, 0, 1.0)',
                             marker_size=tdf0["valid"]*marker_size, marker_symbol=marker_symbol,
                             marker=dict(line=dict(color='black', width=1.5), opacity=opacity, color=marker_color),
                             showlegend=showlegend,
                             customdata  = np.stack((tdf0['vehicle_id'], tdf0['stop_name'], tdf0[data_column], tdf0['direction']), axis=-1),
                             hoverlabel = dict(bgcolor=hover_bgcolor),
                             hovertemplate = ('<i>Vehicle ID</i>: %{customdata[0]}'+\
                                              '<br><b>Stop Name</b>: %{customdata[1]}'+\
                                              '<br><b>Data</b>: %{customdata[2]}'+\
                                              '<br><b>Direction</b>: %{customdata[3]}'+\
                                             '<br><b>Time</b>: %{x|%H:%M:%S}<br><extra></extra>')))
    ############################### FROM DOWNTOWN ###############################
    tdf1 = df[df['plot_no'] == 1].reset_index(drop=True)
    tdf1['direction'] = "→To Downtown" if (tdf1['gtfs_direction_id'] == 0).any() else "←From Downtown"
    fig.add_trace(go.Scatter(x=tdf1['arrival_time'], y=tdf1['stop_sequence'],
                        mode='markers',
                        name=f"right {legend_name}",
                        yaxis="y2", opacity=opacity, fillcolor='rgba(0, 0, 0, 1.0)',
                        marker_size=tdf1["valid"]*marker_size, marker_symbol=marker_symbol,
                        marker=dict(line=dict(color='black', width=1.5), opacity=opacity, color=marker_color),
                        showlegend=showlegend,
                        customdata  = np.stack((tdf1['vehicle_id'], tdf1['stop_name'], tdf1[data_column], tdf1['direction']), axis=-1),
                        hoverlabel = dict(bgcolor=hover_bgcolor),
                        hovertemplate = ('<i>Vehicle ID</i>: %{customdata[0]}'+\
                                        '<br><b>Stop Name</b>: %{customdata[1]}'+\
                                        '<br><b>Data</b>: %{customdata[2]}'+\
                                        '<br><b>Direction</b>: %{customdata[3]}'+\
                                        '<br><b>Time</b>: %{x|%H:%M:%S}<br><extra></extra>')), 
            secondary_y=True)