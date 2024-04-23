import builtins
import datetime

# Global variable to store debug mode
DEBUG_PRINTING = False


def debug_print(*args, **kwargs):
    if DEBUG_PRINTING:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        builtins.print(f"{now}: ", *args, **kwargs)


def timestamped_print(*args, **kwargs):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    builtins.print(f"{now}: ", *args, **kwargs)
