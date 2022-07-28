import sys
import pandas as pd
import os
import datetime as dt
import pickle
import importlib
from pyspark import SparkContext,SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import Row, SparkSession
from pyspark.sql.window import Window
from pyspark import SparkConf
from pyspark.sql.types import TimestampType, DateType,DoubleType,FloatType,IntegerType,StringType,StructType,ArrayType,StructField
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from scipy.stats import zscore
from pyspark.sql.types import IntegerType, StringType, FloatType
import numpy as np
# import gtfs_kit as gk
import time
import math
from copy import deepcopy

from transit_plot_app.src import config

pd.set_option('display.max_columns', None)
import pyspark
print(pyspark.__version__)
spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '80g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '40g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
# recalculate load
def get_derived_load(stops_in):
    stops = stops_in.sort_values(by=['scheduled_time'])
    stops = stops.iloc[1:len(stops)-1]
    last_load = stops.iloc[0]['load']
    derived_load = [last_load]
    for k in range(1, len(stops)):
        cur_stop = stops.iloc[k]
        cur_load = last_load + cur_stop['ons'] - cur_stop['offs']
        derived_load.append(cur_load)
        last_load = cur_load
    stops['derived_load'] = derived_load
    return stops

def timestr_to_seconds(timestr):
    temp = [int(x) for x in timestr.split(":")]
    hour, minute, second = temp[0], temp[1], temp[2]
    return second + minute*60 + hour*3600

def timestr_to_hour(timestr):
    temp = [int(x) for x in timestr.split(":")]
    hour, minute, second = temp[0], temp[1], temp[2]
    return hour

def get_days_of_week(week_arr):
    daysofweek = []
    if week_arr[0] == 1:
        daysofweek.append(1)
    if week_arr[1] == 1:
        daysofweek.append(2)
    if week_arr[2] == 1:
        daysofweek.append(3)
    if week_arr[3] == 1:
        daysofweek.append(4)
    if week_arr[4] == 1:
        daysofweek.append(5)
    if week_arr[5] == 1:
        daysofweek.append(6)
    if week_arr[6] == 1:
        daysofweek.append(7)
    return daysofweek

def seconds_to_timestr(seconds, format='%H:%M:%S'):
    return time.strftime(format, time.gmtime(seconds))
# APC data
# load the APC data

'''Enter path name to APC data'''
filepath = os.path.join(os.getcwd(), "data", config.CARTA_PARQUET_PREPROCESS)
apcdata = spark.read.load(filepath)

# add day and hour of day
apcdata = apcdata.withColumn('day', F.dayofmonth(apcdata.transit_date))
apcdata = apcdata.withColumn('hour', F.hour(apcdata.time_actual_arrive))
apcdata = apcdata.withColumn('dayofweek', F.dayofweek(apcdata.transit_date)) # 1=Sunday, 2=Monday ... 7=Saturday
apcdata.createOrReplaceTempView("apc")

query = f"""
SELECT *
FROM apc
"""
# WHERE (transit_date >= '{date_range[0]} 00:00:00') AND (transit_date < '{date_range[1]} 00:00:00')
apcdata = spark.sql(query)

# remove bad trips
# todelete = apcdata.filter('(ons IS NULL) OR (offs IS NULL) OR (load IS NULL)').select('transit_date','trip_id','overload_id').distinct()
todelete = apcdata.filter('(ons IS NULL) OR (offs IS NULL) OR (load IS NULL)').select('transit_date','trip_id').distinct()
todelete=todelete.withColumn('marker',F.lit(1))
# apcdata=apcdata.join(todelete,on=['transit_date', 'trip_id', 'overload_id'], how='left').filter('marker is null').drop('marker')
apcdata=apcdata.join(todelete,on=['transit_date', 'trip_id'], how='left').filter('marker is null').drop('marker')

# remove trips with less then 5 stops
# todelete = apcdata.groupby('transit_date', 'trip_id', 'overload_id').count().filter("count < 4")
todelete = apcdata.groupby('transit_date', 'trip_id').count().filter("count < 4")
todelete=todelete.withColumn('marker',F.lit(1))
# apcdata=apcdata.join(todelete,on=['transit_date', 'trip_id', 'overload_id'], how='left').filter('marker is null').drop('marker')
apcdata=apcdata.join(todelete,on=['transit_date', 'trip_id'], how='left').filter('marker is null').drop('marker')
apcdata = apcdata.withColumn("route_id_dir", F.concat(F.col('route_id'),F.lit('_'), F.col('route_direction_name')))

## Headway
apcdata = apcdata.withColumn("time_actual_delay", F.col("time_scheduled").cast("long") - F.col("time_actual_arrive").cast("long")) # calculated in seconds
apcdata = apcdata.withColumn("dwell_time", F.col("time_actual_depart").cast("long") - F.col("time_actual_arrive").cast("long")) # calculated in seconds
windowSpec_sched = Window.partitionBy( "transit_date", "route_id", "route_direction_name", "stop_id").orderBy("time_scheduled")
apcdata = apcdata.withColumn("prev_sched", F.lag("time_scheduled", 1).over(windowSpec_sched))
apcdata = apcdata.withColumn("sched_hdwy", F.col("time_scheduled").cast("long") - F.col("prev_sched").cast("long"))
windowSpec_actual = Window.partitionBy( "transit_date", "route_id", "route_direction_name", "stop_id").orderBy("time_actual_depart")
apcdata = apcdata.withColumn("prev_depart", F.lag("time_actual_depart", 1).over(windowSpec_actual))
apcdata = apcdata.withColumn("actual_hdwy", F.col("time_actual_depart").cast("long") - F.col("prev_depart").cast("long"))
apcdata = apcdata.withColumn('is_gapped', F.when(((F.col('actual_hdwy') - F.col('sched_hdwy')) > 0) & ((F.col('actual_hdwy') / F.col('sched_hdwy')) >= 1.5), 1).otherwise(0))
apcdata = apcdata.withColumn('is_bunched', F.when(((F.col('actual_hdwy') - F.col('sched_hdwy')) < 0) & ((F.col('actual_hdwy') / F.col('sched_hdwy')) >= 0) & ((F.col('actual_hdwy') / F.col('sched_hdwy')) <= 0.5), 1).otherwise(0))
apcdata = apcdata.withColumn('is_target', F.when(((F.col('actual_hdwy') / F.col('sched_hdwy')) > 0.5) & ((F.col('actual_hdwy') / F.col('sched_hdwy')) < 1.5), 1).otherwise(0))
# apcdata.select('transit_date','year','month','hour','dayofweek',"route_id",'trip_id','stop_id','block_number','route_direction_name','dwell_time','load','offs','ons','time_actual_arrive','time_actual_depart','delay','time_scheduled','vehicle_id','stop_name','actual_hdwy','prev_sched').filter()
# os.getcwd()

apcdata.write.partitionBy("year", 'month').mode("overwrite").parquet('./transit_plot_app/data/{0}'.format(config.CARTA_PARQUET)
