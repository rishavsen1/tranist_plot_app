# Contents of ~/my_app/streamlit_app.py
import streamlit as st
import datetime as dt
import os
from src import plot_utils
from src import data_utils
from src import config
from pyspark.sql import SparkSession

st.markdown("# Transit Max Occupancy Heatmap")
st.sidebar.markdown("# Parameters")

spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '40g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '20g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .getOrCreate()

with st.sidebar:
    st.write("Use case box not setup yet.")
    use_case_selectbox = st.selectbox('Use case', ('Historical', 'Prediction'))
    dataset_selectbox = st.selectbox('Dataset', ('Chattanooga, CARTA', 'Nashville, MTA'))
    
    time_granularity = st.number_input('Aggregation time (in minutes)', value=60)
    
    if dataset_selectbox == 'Nashville, MTA':
        filter_date = st.date_input('Filter dates', min_value=dt.date(2020, 1, 1), max_value=dt.date(2022, 4, 6),
                                    value=(dt.date(2021, 10, 18), dt.date(2021, 10, 19)))
    elif dataset_selectbox == 'Chattanooga, CARTA':
        filter_date = st.date_input('Filter dates', min_value=dt.date(2019, 1, 1), max_value=dt.date(2022, 5, 30),
                                    value=(dt.date(2021, 10, 18), dt.date(2021, 10, 19)))
    else:
        filter_date = st.date_input('Filter dates', min_value=dt.date(2020, 1, 1), max_value=dt.date(2022, 4, 6),
                                    value=(dt.date(2021, 10, 18), dt.date(2021, 10, 19)))
        
    plot_button = st.button('Plot graph')
    
def route_dir_func(row):
    return (row.route_id + row.row.route_direction_name)

if plot_button:
    if dataset_selectbox == 'Nashville, MTA':
        filepath = os.path.join(os.getcwd(), "data", config.MTA_PARQUET)
    elif dataset_selectbox == 'Chattanooga, CARTA':
        filepath = os.path.join(os.getcwd(), "data", config.CARTA_PARQUET)
    else:
        st.error("Select dataset")
    start = filter_date[0].strftime('%Y-%m-%d')
    end   = filter_date[1].strftime('%Y-%m-%d')
    
    with st.spinner(f"Loading {dataset_selectbox} files..."):
        apcdata = spark.read.load(filepath)
        apcdata.createOrReplaceTempView("apc")
        
        # filter subset
        query = f"""
                SELECT *
                FROM apc
                WHERE (transit_date >= '{start}') AND (transit_date <= '{end}')
                """
        
        if dataset_selectbox == 'Nashville, MTA':
            apcdata=spark.sql(query)
            apcdata = data_utils.remove_nulls_from_apc(apcdata)
            apcdata.createOrReplaceTempView('apcdata')
            apcdata_per_trip = data_utils.get_apc_per_trip_sparkview(spark)
            apcdata_per_trip = apcdata_per_trip.withColumnRenamed("route_id_direction","route_id_dir")
            apcdata_per_trip = apcdata_per_trip.drop("load")
            df = apcdata_per_trip.toPandas()
        elif dataset_selectbox == 'Chattanooga, CARTA':
            apcdata_per_trip=spark.sql(query)
            apcdata_per_trip = apcdata_per_trip.na.drop(subset=["time_actual_arrive"])
            df = apcdata_per_trip.toPandas()

        st.dataframe(df.head())
        st.dataframe(df.tail())
        
    with st.spinner(f"Preparing {dataset_selectbox} graphs..."):
        fig = plot_utils.plot_max_aggregate(df, filter_date[0], None, time_granularity, dataset_selectbox)
        st.plotly_chart(fig, use_container_width=True)