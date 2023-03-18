import time

from rich.live import Live
from rich.table import Table

table = Table()
table.add_column("Row ID")
table.add_column("Description")
table.add_column("Level")

with Live(table, refresh_per_second=4) as live:  # update 4 times a second to feel fluid
    for row in range(12):
        live.console.print(f"Working on row #{row}")
        time.sleep(0.4)
        table.add_row(f"{row}", f"description {row}", "[red]ERROR")
        
# pool = ThreadPool(daemon = True)
# def watchprocess():
#     while True:
#         length = len(threading.enumerate())
#         print('当前运行的线程数为：%d'%length)
#         if length<=1:
#             break
# watchprocess = threading.Thread(watchprocess(),daemon=True)