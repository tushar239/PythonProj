"""
http://www.tutorialspoint.com/python/python_date_time.htm

Modules
    - time
    - calendar

In Python, time can be retrieved in Seconds, TimeTuple, Formatted Time
Python also provides a capability of displaying Calendar.
"""

import time, calendar

timeInSeconds = time.time()
print("Time in seconds: "+ str(timeInSeconds)) # Time in seconds: 1464043435.923485

# converting seconds to time-tuple
# TimeTuple is nothing but just a record of time related attributes like tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec etc.
timeTupleOfLocalTime = time.localtime(timeInSeconds)
print("Time Tuple: "+ str(timeTupleOfLocalTime)) # Time Tuple: time.struct_time(tm_year=2016, tm_mon=5, tm_mday=23, tm_hour=15, tm_min=45, tm_sec=36, tm_wday=0, tm_yday=144, tm_isdst=1)

# time.ctime(seconds) function
print("ctime(seconds) is same as asktime(timeTuple)" + str(time.ctime(timeInSeconds))) # Mon May 23 15:52:49 2016

# time.gmtime(seconds) give UTC time
print("UTC TimeTuple: "+ str(time.gmtime(timeInSeconds))) # UTC time: time.struct_time(tm_year=2016, tm_mon=5, tm_mday=23, tm_hour=23, tm_min=3, tm_sec=28, tm_wday=0, tm_yday=144, tm_isdst=0)

# Formatting TimeTuple
formattedTime = time.asctime(timeTupleOfLocalTime)
print("Formatted Time: "+formattedTime) # Formatted Time: Mon May 23 15:50:51 2016

# time.clock() returns the current CPU time as a floating-point number of seconds. To measure computational costs of different approaches, the value of time.clock is more useful than that of time.time().
print("CPU time: "+ str(time.clock()))

# time.strptime(time in some format, format) will return a tuple of TimeTuple attributes
# http://www.tutorialspoint.com/python/time_strptime.htm
struct_time = time.strptime("30 Nov 00", "%d %b %y")
print("returned tuple: %s" % str(struct_time)) # returned tuple: time.struct_time(tm_year=2000, tm_mon=11, tm_mday=30, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=3, tm_yday=335, tm_isdst=-1)


# calendar is module in Python that you can use to display a calendar
print("Calendar:\n"+calendar.month(2016, 1))
"""
O/P:

    January 2016
Mo Tu We Th Fr Sa Su
             1  2  3
 4  5  6  7  8  9 10
11 12 13 14 15 16 17
18 19 20 21 22 23 24
25 26 27 28 29 30 31
"""


