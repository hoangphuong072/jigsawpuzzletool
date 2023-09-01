import concurrent.futures

from LinePrint import line_print as print,start_log_by_pid,end_log_by_pid

def worker():
    start_log_by_pid()
    print("Dsadsa")
    end_log_by_pid()
with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
    executor.submit(worker)
