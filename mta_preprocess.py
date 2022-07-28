import os
os.chdir("..")
print(os.getcwd())
import sys
import pandas as pd
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
import gtfs_kit as gk
import time
import math
from src import config

pd.set_option('display.max_columns', None)
import pyspark
print(pyspark.__version__)
spark = SparkSession.builder.config('spark.executor.cores', '8').config('spark.executor.memory', '40g')\
        .config("spark.sql.session.timeZone", "UTC").config('spark.driver.memory', '20g').master("local[26]")\
        .appName("wego-daily").config('spark.driver.extraJavaOptions', '-Duser.timezone=UTC').config('spark.executor.extraJavaOptions', '-Duser.timezone=UTC')\
        .config("spark.sql.datetime.java8API.enabled", "true").config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

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

filepath = os.path.join(os.getcwd(), "data", config.MTA_PARQUET_PREPROCESS)
apcdata = spark.read.load(filepath)

# add day and hour of day
apcdata = apcdata.withColumn('day', F.dayofmonth(apcdata.transit_date))
apcdata = apcdata.withColumn('hour', F.hour(apcdata.arrival_time))
apcdata = apcdata.withColumn('dayofweek', F.dayofweek(apcdata.transit_date)) # 1=Sunday, 2=Monday ... 7=Saturday
apcdata.createOrReplaceTempView("apc")

query = f"""
SELECT *
FROM apc
"""
apcdata = spark.sql(query)

# remove bad trips
todelete = apcdata.filter('(ons IS NULL) OR (offs IS NULL) OR (load IS NULL)').select('transit_date','trip_id','overload_id').distinct()
todelete=todelete.withColumn('marker',F.lit(1))
apcdata=apcdata.join(todelete,on=['transit_date', 'trip_id', 'overload_id'], how='left').filter('marker is null').drop('marker')

# remove trips with less then 5 stops
todelete = apcdata.groupby('transit_date', 'trip_id', 'overload_id').count().filter("count < 4")
todelete=todelete.withColumn('marker',F.lit(1))
apcdata=apcdata.join(todelete,on=['transit_date', 'trip_id', 'overload_id'], how='left').filter('marker is null').drop('marker')

# # Weather - darksky
# filepath = os.path.join(os.getcwd(), "data", "weather", "darksky_nashville_20220406.csv")
# darksky = pd.read_csv(filepath)
# # GMT-5
# darksky['datetime'] = darksky['time'] - 18000
# darksky['datetime'] = pd.to_datetime(darksky['datetime'], infer_datetime_format=True, unit='s')
# darksky = darksky.set_index(darksky['datetime'])
# # darksky = darksky.sort_index().loc[date_range[0]:date_range[1]]
# darksky['year'] = darksky['datetime'].dt.year
# darksky['month'] = darksky['datetime'].dt.month
# darksky['day'] = darksky['datetime'].dt.day
# darksky['hour'] = darksky['datetime'].dt.hour
# val_cols= ['temperature', 'humidity', 'nearest_storm_distance', 'precipitation_intensity', 'precipitation_probability', 'pressure', 'wind_gust', 'wind_speed']
# join_cols = ['year', 'month', 'day', 'hour']
# darksky = darksky[val_cols+join_cols]
# renamed_cols = {k: f"darksky_{k}" for k in val_cols}
# darksky = darksky.rename(columns=renamed_cols)
# darksky = darksky.groupby(['year', 'month', 'day', 'hour']).mean().reset_index()
# darksky=spark.createDataFrame(darksky)
# darksky.createOrReplaceTempView("darksky")

# join apc and darksky
# apcdata = apcdata.join(darksky,on=['year', 'month', 'day', 'hour'], how='left')
# # Weather - weatherbit
# # load weatherbit
# filepath = os.path.join(os.getcwd(), "data", "weather", "weatherbit_weather_2010_2022.parquet")
# weatherbit = spark.read.load(filepath)

# weatherbit = weatherbit.filter("(spatial_id = 'Berry Hill') OR (spatial_id = 'Belle Meade')")
# weatherbit.createOrReplaceTempView("weatherbit")
# query = f"""
# SELECT *
# FROM weatherbit
# """
# # WHERE (timestamp_local >= '{date_range[0]} 23:00:00') AND (timestamp_local < '{date_range[1]} 00:00:00')
# weatherbit = spark.sql(query)

# weatherbit = weatherbit.withColumn('year', F.year(weatherbit.timestamp_local))
# weatherbit = weatherbit.withColumn('month', F.month(weatherbit.timestamp_local))
# weatherbit = weatherbit.withColumn('day', F.dayofmonth(weatherbit.timestamp_local))
# weatherbit = weatherbit.withColumn('hour', F.hour(weatherbit.timestamp_local))
# weatherbit = weatherbit.select('year', 'month', 'day', 'hour', 'rh', 'wind_spd', 'slp', 'app_temp', 'temp', 'snow', 'precip')
# weatherbit = weatherbit.groupBy('year', 'month', 'day', 'hour').agg(F.mean('rh').alias('weatherbit_rh'), \
#                                                                     F.mean('wind_spd').alias('weatherbit_wind_spd'), \
#                                                                     F.mean('app_temp').alias('weatherbit_app_temp'), \
#                                                                     F.mean('temp').alias('weatherbit_temp'), \
#                                                                     F.mean('snow').alias('weatherbit_snow'), \
#                                                                     F.mean('precip').alias('weatherbit_precip')
#                                                                    )
# weatherbit = weatherbit.sort(['year', 'month', 'day', 'hour'])

# join apc and weatherbit
# apcdata=apcdata.join(weatherbit,on=['year', 'month', 'day', 'hour'], how='left')
# # Join with GTFS
# filepath = os.path.join(os.getcwd(), "data", "static_gtfs", "alltrips_mta_wego.parquet")
# alltrips = spark.read.load(filepath)

# add gtfs_file, gtfs_shape_id, gtfs_route_id, gtfs_direction_id, gtfs_start_date, gtfs_end_date, gtfs_date
# gtfstrips = alltrips.select('trip_id','date','gtfs_file', 'shape_id', 'route_id', 'direction_id', 'start_date', 'end_date').distinct()
# gtfstrips = gtfstrips.withColumnRenamed('shape_id', 'gtfs_shape_id')\
#                      .withColumnRenamed('route_id', 'gtfs_route_id')\
#                      .withColumnRenamed('direction_id', 'gtfs_direction_id')\
#                      .withColumnRenamed('start_date','gtfs_start_date')\
#                      .withColumnRenamed('end_date','gtfs_end_date')

# Some GTFS are outdated?, add transit_date, and trip_id
# rantrips = apcdata.select('transit_date','trip_id').distinct().join(gtfstrips, on='trip_id', how='left').filter('transit_date >= date')
# rantrips_best_gtfs_file = rantrips.groupby('transit_date','trip_id').agg(F.max('date').alias('date'))
# # Inner assures no NaN
# rantrips = rantrips.join(rantrips_best_gtfs_file, on=['transit_date','trip_id','date'], how='inner').withColumnRenamed('date', 'gtfs_date')
# # Essentilly rantrips is just the GTFS data with transit_id and trip_id (matched from the apcdata)
# apcdata = apcdata.join(rantrips,on=['transit_date','trip_id'], how='left')
# # get scheduled number of vehicles on route at the given hour

# alltrips = alltrips.withColumnRenamed('route_id', 'gtfs_route_id')\
#                    .withColumnRenamed('date', 'gtfs_date')\
#                    .withColumnRenamed('direction_id', 'gtfs_direction_id')

# timestrToSecondsUDF = F.udf(lambda x: timestr_to_seconds(x), IntegerType())
# alltrips = alltrips.withColumn("time_seconds", timestrToSecondsUDF(F.col('arrival_time')))

# timestrToHourUDF = F.udf(lambda x: timestr_to_hour(x), IntegerType())
# alltrips = alltrips.withColumn("hour", timestrToHourUDF(F.col('arrival_time')))

# getDaysOfWeekUDF = F.udf(lambda x: get_days_of_week(x), ArrayType(IntegerType()))
# alltrips = alltrips.withColumn("dayofweek", getDaysOfWeekUDF(F.array('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday')))
# alltrips = alltrips.withColumn("dayofweek", F.explode("dayofweek"))

# alltrips.createOrReplaceTempView('alltrips_1')

# query = """
# SELECT gtfs_date, dayofweek, hour, gtfs_route_id, gtfs_direction_id, count(trip_id) AS gtfs_number_of_scheduled_trips
# FROM alltrips_1
# GROUP BY gtfs_date, dayofweek, hour, gtfs_route_id, gtfs_direction_id
# """
# trips_per_route = spark.sql(query)
# apcdata = apcdata.join(trips_per_route, on=['gtfs_date', 'dayofweek', 'hour', 'gtfs_route_id', 'gtfs_direction_id'], how='left')
# # get scheduled trips per stop

# query = """
# SELECT gtfs_date, dayofweek, hour, gtfs_route_id, gtfs_direction_id, stop_id, count(trip_id) AS gtfs_number_of_scheduled_trips_at_stop
# FROM alltrips_1
# GROUP BY gtfs_date, dayofweek, hour, gtfs_route_id, gtfs_direction_id, stop_id
# """
# trips_per_stop = spark.sql(query)
# apcdata = apcdata.join(trips_per_stop, on=['gtfs_date', 'dayofweek', 'hour', 'gtfs_route_id', 'gtfs_direction_id', 'stop_id'], how='left')

## Headway
apcdata = apcdata.withColumn("delay_time", F.col("scheduled_time").cast("long") - F.col("arrival_time").cast("long")) # calculated in seconds
apcdata = apcdata.withColumn("dwell_time", F.col("departure_time").cast("long") - F.col("arrival_time").cast("long")) # calculated in seconds
windowSpec_sched = Window.partitionBy( "transit_date", "route_id", "route_direction_name", "stop_id").orderBy("scheduled_time")
apcdata = apcdata.withColumn("prev_sched", F.lag("scheduled_time", 1).over(windowSpec_sched))
apcdata = apcdata.withColumn("sched_hdwy", F.col("scheduled_time").cast("long") - F.col("prev_sched").cast("long"))
windowSpec_actual = Window.partitionBy( "transit_date", "route_id", "route_direction_name", "stop_id").orderBy("departure_time")
apcdata = apcdata.withColumn("prev_depart", F.lag("departure_time", 1).over(windowSpec_actual))
apcdata = apcdata.withColumn("actual_hdwy", F.col("departure_time").cast("long") - F.col("prev_depart").cast("long"))
apcdata = apcdata.withColumn('is_gapped', F.when(((F.col('actual_hdwy') - F.col('sched_hdwy')) > 0) & ((F.col('actual_hdwy') / F.col('sched_hdwy')) >= 1.5), 1).otherwise(0))
apcdata = apcdata.withColumn('is_bunched', F.when(((F.col('actual_hdwy') - F.col('sched_hdwy')) < 0) & ((F.col('actual_hdwy') / F.col('sched_hdwy')) >= 0) & ((F.col('actual_hdwy') / F.col('sched_hdwy')) <= 0.5), 1).otherwise(0))
apcdata = apcdata.withColumn('is_target', F.when(((F.col('actual_hdwy') / F.col('sched_hdwy')) > 0.5) & ((F.col('actual_hdwy') / F.col('sched_hdwy')) < 1.5), 1).otherwise(0))

duplicates=apcdata.groupby(['transit_date','trip_id','route_id','route_direction_name','stop_id_original', 'stop_sequence','block_abbr','vehicle_id']).count()
todelete=duplicates.filter('count >1').select('transit_date','block_abbr').distinct()
todelete=todelete.withColumn('indicator',F.lit(1))

# for null vehicle id -- remove the whole block
nullvehicleids=apcdata.filter('vehicle_id="NULL" or vehicle_id is null').select('transit_date','block_abbr').distinct()
nullvehicleids=nullvehicleids.withColumn('indicator',F.lit(1))
nullvehicleids.count()

null_arrival_departure_times=apcdata.groupBy('transit_date', 'trip_id','vehicle_id','overload_id','block_abbr')  .agg((F.sum(F.col('arrival_time').isNull().cast("int")).alias('null_arrival_count')),F.count('*').alias('total_count'))
null_arrival_departure_times=null_arrival_departure_times.filter('null_arrival_count = total_count').select('transit_date','block_abbr').distinct()
null_arrival_departure_times=null_arrival_departure_times.withColumn('indicator',F.lit(1))
null_arrival_departure_times.count()

apcdata.show(5)
f = os.path.join(os.getcwd(), 'data', 'mta_apc_out.parquet')
apcdata.write.partitionBy("year", 'month').mode("overwrite").parquet(f)