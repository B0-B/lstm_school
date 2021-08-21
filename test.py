from datetime import datetime, timedelta
Days = 4
dt = datetime.now()-timedelta(days=1)
min_now = str(int(int(dt.strftime("%M"))/5)*5)
if len(min_now) < 2: min_now = "0" + min_now
t_now = f'''{dt.strftime("%H")}:{min_now} {dt.strftime("%m-%d-%y")}'''
t = datetime.strptime(t_now, "%H:%M %m-%d-%y") - timedelta(days=Days)
t_then = t.strftime("%H:%M %m-%d-%y")