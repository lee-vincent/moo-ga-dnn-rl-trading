import builtins
import datetime


def timestamped_print(*args, **kwargs):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    builtins.print(f"{now}: ", *args, **kwargs)
