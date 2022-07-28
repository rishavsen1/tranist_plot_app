import streamlit as st
import plotly.express as px
import datetime as dt
import pandas as pd
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F
from src import config

spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '40g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '20g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .getOrCreate()

st.markdown("# Route-wise Statistics")
st.sidebar.markdown("# Per Route-wise Statistics")

tab_occupancy, tab_boarding, tab_headway, tab_delay, tab_stops = st.tabs(['Occupancy', 'Boardings', 'Headways', 'Delays', 'Stops'])

# Everytime you change something here, the entire site will refresh.
with st.sidebar:
    dataset_selectbox = st.selectbox('Dataset', ('Chattanooga, CARTA', 'Nashville, MTA'))
    if dataset_selectbox == 'Chattanooga, CARTA':
        fp = os.path.join('data', 'CARTA_route_ids.csv')
        filter_date = st.date_input('Filter dates', min_value=dt.date(2019, 1, 1), max_value=dt.date(2022, 5, 30),
                                    value=(dt.date(2021, 10, 18), dt.date(2021, 10, 19)))
    elif dataset_selectbox == 'Nashville, MTA':
        fp = os.path.join('data', 'MTA_route_ids.csv')
        filter_date = st.date_input('Filter dates', min_value=dt.date(2020, 1, 1), max_value=dt.date(2022, 4, 6),
                                    value=(dt.date(2021, 10, 18), dt.date(2021, 10, 19)))
    
    route_list = pd.read_csv(fp).dropna().sort_values('route_id').route_id.tolist()
    route_option = st.selectbox('Route:', route_list)
    
    agg_time = st.number_input('Aggregation window (mins):', value=15)
    window = agg_time

    points = st.selectbox('Show box plot points:', (False, 'all', 'outliers', 'suspectedoutliers'))
    plot_button = st.button('Plot graphs')
    
if plot_button:
    if dataset_selectbox == 'Nashville, MTA':
        filepath = os.path.join(os.getcwd(), "data", config.MTA_PARQUET)
    elif dataset_selectbox == 'Chattanooga, CARTA':
        filepath = os.path.join(os.getcwd(), "data", config.CARTA_PARQUET)
    else:
        st.error("Select dataset")

    start = filter_date[0].strftime('%Y-%m-%d')
    end   = filter_date[1].strftime('%Y-%m-%d')
    
    time_range = pd.date_range(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-02'), freq=f'{agg_time}min').time
    
    with st.spinner(f"Loading {dataset_selectbox} files..."):
        st.write(f"Plots of {dataset_selectbox} using {agg_time} minute time windows.")

        if dataset_selectbox == 'Nashville, MTA':
            st.write(filepath)
            apcdata = spark.read.load(filepath)
            apcdata.createOrReplaceTempView("apc")

            # filter subset
            query = """
                    SELECT transit_date, trip_id, 
                           first(arrival_time) as arrival_time,
                           first(route_id) AS route_id, 
                           first(hour) as hour,
                           first(route_direction_name) AS route_direction_name, 
                           percentile(INT(delay_time), 0.95) AS delay,
                           percentile(INT(actual_hdwy), 0.95) AS headway,
                           percentile(INT(load), 1.00) AS y_reg100,
                           percentile(INT(ons), 1.00) AS boardings
                    FROM apc
                    GROUP BY transit_date, trip_id
                    ORDER BY arrival_time
                    """
            apcdata_per_trip=spark.sql(query)
            apcdata_per_trip = apcdata_per_trip.where(apcdata_per_trip.route_id == route_option)
            apcdata_per_trip = apcdata_per_trip.filter(F.col("transit_date").between(filter_date[0], filter_date[1]))
            apcdata_per_trip = apcdata_per_trip.withColumn("minute", F.minute("arrival_time"))
            apcdata_per_trip = apcdata_per_trip.withColumn("minuteByWindow", apcdata_per_trip.minute/window)
            apcdata_per_trip = apcdata_per_trip.withColumn("time_window", apcdata_per_trip.minuteByWindow + (apcdata_per_trip.hour * (60 / window)))
            apcdata_per_trip = apcdata_per_trip.withColumn("time_window", F.floor(apcdata_per_trip.time_window).cast(IntegerType()))
            apcdata_per_trip = apcdata_per_trip.filter(apcdata_per_trip.y_reg100 <= 100.0)
            
            df = apcdata_per_trip.toPandas()
            df = df.rename({'route_direction_name':'direction'}, axis=1)
            df = df.rename({'y_reg100':'load'}, axis=1)
            df = df.dropna()

        elif dataset_selectbox == 'Chattanooga, CARTA':
            st.write(filepath)
            apcdata = spark.read.load(filepath)
            apcdata.createOrReplaceTempView("apc")

            # filter subset
            query = """
                    SELECT transit_date, trip_id, 
                           first(time_actual_arrive) as time_actual_arrive,
                           first(route_id) AS route_id, 
                           first(hour) as hour,
                           first(route_direction_name) AS direction, 
                           percentile(INT(delay), 0.95) AS delay,
                           percentile(INT(actual_hdwy), 0.95) AS headway,
                           percentile(INT(load), 1.00) AS load,
                           percentile(INT(ons), 1.00) AS boardings
                    FROM apc
                    GROUP BY transit_date, trip_id
                    ORDER BY time_actual_arrive
                    """
            apcdata=spark.sql(query)
            apcdata = apcdata.where(apcdata.route_id == route_option)
            apcdata = apcdata.filter(F.col("transit_date").between(filter_date[0], filter_date[1]))
            apcdata = apcdata.na.drop(subset=["time_actual_arrive"])
            apcdata = apcdata.withColumnRenamed("route_direction_name", "direction")
            apcdata = apcdata.withColumnRenamed("ons", "boardings")
            apcdata = apcdata.withColumn("minute", F.minute("time_actual_arrive"))
            apcdata = apcdata.withColumn("minuteByWindow", apcdata.minute/window)
            apcdata = apcdata.withColumn("time_window", apcdata.minuteByWindow + (apcdata.hour * (60 / window)))
            apcdata = apcdata.withColumn("time_window", F.floor(apcdata.time_window).cast(IntegerType()))
            
            df = apcdata.toPandas()
        
        # Common code
        df['time_window'] = df['time_window'].astype('int')
        df['time_window_str'] = df['time_window'].apply(lambda x: time_range[x])
        
        st.dataframe(df.head())
        st.dataframe(df.tail())
        if df.empty:
            st.error("Dataframe is empty.")
            
        # 1. Loads box plot
        with tab_occupancy:
            st.write(f"max occupancy in {agg_time} minute windows")
            fig = px.box(df, x="time_window_str", y="load", facet_row="direction", color="direction",
                        boxmode="overlay", points=points)
            fig.update_xaxes(tickformat="%H:%M")
            layout = fig.update_layout(
                title='Max Occupancy',
                xaxis=dict(title='Aggregation interval (hh:mm)', tickformat="%H:%M")
            )
            for a in fig.layout.annotations:
                a.text = a.text.split("=")[1]
            st.plotly_chart(fig)
        
        # 2. Boardings box plot
        with tab_boarding:
            st.write(f"boarding events in {agg_time} minute windows (scatter plot)")
            fig = px.box(df, x="time_window_str", y="boardings", facet_row="direction", color="direction",
                            boxmode="overlay", points=points)
            fig.update_xaxes(tickformat="%H:%M")
            layout = fig.update_layout(
                title='Boardings',
                xaxis=dict(title='Aggregation interval (hh:mm)', tickformat="%H:%M")
            )
            for a in fig.layout.annotations:
                a.text = a.text.split("=")[1]
            st.plotly_chart(fig)
        
        # 3. Actual headways (limit to less than 5 hours)
        with tab_headway:
            st.write(f"headway (average gap between trips) in {agg_time} minute windows")
            fig = px.box(df[df['headway'] <= 3 * 3600], x="time_window_str", y="headway", facet_row="direction", color='direction',
                            boxmode="overlay", points=points)
            fig.update_xaxes(tickformat="%H:%M")
            # fig.update_xaxes(tickformat="%H:%M", dtick=agg_time * 60 * 5)
            fig.update_yaxes(title='Headway (s)')
            layout = fig.update_layout(
                title='Actual headways',
                xaxis=dict(title='Aggregation interval (hh:mm)', tickformat="%H:%M"),
                # width=1000, height=500,
                yaxis_range=[-400, 10000]
            )
            for a in fig.layout.annotations:
                a.text = a.text.split("=")[1]
            st.plotly_chart(fig)
        
        # 4. Delays
        with tab_delay:
            st.write(f"delays (scheduled vs actual time at the arrival at a stop) in {agg_time} minute windows")
            fig = px.box(df[(df.delay > -300) & (df.delay < 3 * 3600)], x="time_window_str", y="delay", facet_row="direction", color='direction',
                            boxmode="overlay", points=points)
            fig.update_xaxes(tickformat="%H:%M")
            fig.update_yaxes(title='Delay (s)')
            layout = fig.update_layout(
                title='Delays',
                xaxis=dict(title='Aggregation interval (hh:mm)', tickformat="%H:%M")
            )
            for a in fig.layout.annotations:
                a.text = a.text.split("=")[1]
            st.plotly_chart(fig)

        # 5. 95th percentile of boardings in a stop
        if dataset_selectbox == 'Nashville, MTA':
            apcdata = spark.read.load(filepath)
            apcdata.createOrReplaceTempView("apc")
            # filter subset
            query = f"""
                    SELECT ons, hour, arrival_time, stop_name, stop_sequence, transit_date, route_id, route_direction_name
                    FROM apc
                    WHERE (transit_date >= '{start}') AND (transit_date <= '{end}' AND route_id == '{route_option}')
                    """
            apcdata=spark.sql(query)
            
            window = agg_time # minutes
            apcdata = apcdata.withColumn("minute", F.minute("arrival_time"))
            apcdata = apcdata.withColumn("minuteByWindow", apcdata.minute/window)
            apcdata = apcdata.withColumn("time_window", apcdata.minuteByWindow + (apcdata.hour * (60 / window)))
            apcdata = apcdata.withColumn("time_window", F.floor(apcdata.time_window).cast(IntegerType()))
            apcdata = apcdata.drop("minuteByWindow", "minute", "hour")
            apcdata = apcdata.na.drop(subset=["time_window"])
            df = apcdata.toPandas()
            df['time_window'] = df['time_window'].astype('int')
            df['time_window_str'] = df['time_window'].apply(lambda x: time_range[x])
            df = df.rename({'route_direction_name':'direction'}, axis=1)
            df_stop = df.groupby(['stop_sequence', 'time_window_str']).agg({'ons':'sum', 'direction':'first'}).reset_index()[['stop_sequence', 'time_window_str', 'ons', 'direction']].dropna()
            df_stop = df_stop[df_stop['stop_sequence'] > 1]
            
        elif dataset_selectbox == 'Chattanooga, CARTA':
            apcdata = spark.read.load(filepath)
            apcdata.createOrReplaceTempView("apc")
            # filter subset
            query = f"""
                    SELECT ons, hour, time_actual_arrive, stop_name, transit_date, route_id, route_direction_name
                    FROM apc
                    WHERE (transit_date >= '{start}') AND (transit_date <= '{end}' AND route_id == '{route_option}')
                    ORDER BY time_actual_arrive
                    """
            apcdata=spark.sql(query)
            
            window = agg_time # minutes
            apcdata = apcdata.withColumn("minute", F.minute("time_actual_arrive"))
            apcdata = apcdata.withColumn("minuteByWindow", apcdata.minute/window)
            apcdata = apcdata.withColumn("time_window", apcdata.minuteByWindow + (apcdata.hour * (60 / window)))
            apcdata = apcdata.withColumn("time_window", F.floor(apcdata.time_window).cast(IntegerType()))
            apcdata = apcdata.drop("minuteByWindow", "minute", "hour")
            apcdata = apcdata.na.drop(subset=["time_window"])
            apcdata = apcdata.withColumnRenamed("route_direction_name", "direction")
            df = apcdata.toPandas()
            df['time_window'] = df['time_window'].astype('int')
            df['time_window_str'] = df['time_window'].apply(lambda x: time_range[x])
            df_stop = df.groupby(['stop_name', 'time_window_str']).agg({'ons':'sum', 'direction':'first'}).reset_index()[['stop_name', 'time_window_str', 'ons', 'direction']].dropna()

        # Common code
        with tab_stops:
            df_stop = df_stop.rename({'ons':'boardings'}, axis=1)
            st.write(f"boarding per stop per day in {agg_time} minute windows")
            fig = px.box(df_stop, x="time_window_str", y="boardings", facet_row='direction', color='direction', 
                            boxmode="overlay", points=points)
            fig.update_xaxes(tickformat="%H:%M")
            layout = fig.update_layout(
                title='Boardings per stop per time window',
                xaxis=dict(title='Aggregation interval (hh:mm)', tickformat="%H:%M")
            )
            for a in fig.layout.annotations:
                a.text = a.text.split("=")[1]
            st.plotly_chart(fig)