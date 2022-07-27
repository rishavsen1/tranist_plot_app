import math

def get_time_window(row, window):
    minute = row.arrival_time.minute
    minuteByWindow = minute//window
    temp = minuteByWindow + (row.arrival_time.hour * (60/window))
    return math.floor(temp)