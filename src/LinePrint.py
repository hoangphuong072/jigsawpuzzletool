from inspect import currentframe, getframeinfo


def line_print(log):
    cf = currentframe().f_back
    filename = getframeinfo(cf).filename
    print(f'{filename}:{cf.f_lineno} --> {str(log)}')
