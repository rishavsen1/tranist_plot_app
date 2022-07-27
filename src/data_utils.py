import math

def get_time_window(row, window):
    minute = row.time_actual_arrive.minute
    minuteByWindow = minute//window
    temp = minuteByWindow + (row.time_actual_arrive.hour * (60/window))
    return math.floor(temp)

def get_class(x):
    percentiles = [(0.0, 9.0), (10.0, 15.0), (16.0, 55.0), (56.0, 75.0), (76.0, 100.0)]
    for i, (s, e) in enumerate(percentiles):
        if x >= s*40*0.01 and x <= e*40*0.01:
            print(x, s, e, i)
            return i