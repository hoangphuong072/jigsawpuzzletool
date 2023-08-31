import os
from inspect import currentframe, getframeinfo


def line_print(log):
    cf = currentframe().f_back
    filename = getframeinfo(cf).filename
    pid = os.getpid()

    if pid in line_print_data.keys():
        msg = f'{filename}:{cf.f_lineno} (PID:{pid})--> {str(log)}'
        line_print_data[pid] += msg + "\n"
    else:
        print(f'{filename}:{cf.f_lineno} --> {str(log)}')


line_print_data = {}


# def line_print_multiproccess(log, proccess_name="", display=False, print_all=True, clear=False):
#     pid = os.getpid()
#     cf = currentframe().f_back
#     filename = getframeinfo(cf).filename
#     msg = f'{filename}:{cf.f_lineno} (PID:{pid})--> {str(log)}'
#     if display:
#         print(msg)
#     else:
#         if pid not in line_print_data.keys(): line_print_data[pid] = ''
#         line_print_data[pid] += msg + "\n"
#         if print_all:
#             print(f"--------------|------{proccess_name}------|------------")
#             print(line_print_data[pid])
#         if clear: line_print_data[pid] = ''


def start_log_by_pid():
    pid = os.getpid()
    line_print_data[pid] = ''


def end_log_by_pid():
    pid = os.getpid()
    print(line_print_data[pid])
    line_print_data.pop(pid)
