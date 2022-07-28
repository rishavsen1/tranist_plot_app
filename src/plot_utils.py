import pandas as pd
from src import data_utils
import numpy as np
import plotly.graph_objects as go

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
        
        if dataset_selectbox == 'Nashville, MTA':
            _df['time_window'] = _df.apply(lambda x: data_utils.get_time_window(x, TIME_GRANULARITY, row_name='arrival_time'), axis=1)
            _df = _df.groupby('time_window').max()
            indices = _df.index.astype('int')
            _df['y_classes'] = _df['y_reg100'].apply(lambda x: data_utils.get_class(x, dataset_selectbox))
        elif dataset_selectbox == 'Chattanooga, CARTA':
            _df['time_window'] = _df.apply(lambda x: data_utils.get_time_window(x, TIME_GRANULARITY, row_name='time_actual_arrive'), axis=1)
            _df = _df.groupby('time_window').max()
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
    bvals = [0, 1, 2, 3, 4, 5]
    colors = ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026']
    dcolorsc = discrete_colorscale(bvals, colors)
    tickvals = [0, 1, 2, 3, 4, 5]
    ticktext = ['low', 'med', 'med-high', 'high', 'very-high']
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
        xaxis_tickformat='%H:%M:%S',
        xaxis_hoverformat='%H:%M:%S',
    )

    # fig.write_html(f"plots/{TIME_GRANULARITY}min_window_{DATE_START}.html")
    return fig