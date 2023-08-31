import os
import time
from LinePrint import line_print as print
time_log = {}


def start_tracking_time(name):
    time_log[str(os.getpid())+"-"+name] = time.time()


def end_tracking_time(name,clear=True):
    t = time.time() - time_log[str(os.getpid())+"-"+name]
    if clear:
        time_log.pop(str(os.getpid())+"-"+name)
    return t


def print_tracking_time(name,clear=True):
    print(f"{name} running : {str(end_tracking_time(name,clear=clear))}")

