import os, time
import re
from rich import print
from rich.table import Table
from rich.live import Live
import datetime
from rich.console import Console
import win32ui
import win32gui
console = Console(width=200)


def interactive(func):
    def wrapper(*args):
        try:
            func(*args)
        except KeyboardInterrupt:
            pass
    return wrapper


class ITable(Table):
    def __init__(self, *args, num_columns=None, **kwargs):
        super().__init__(*args, **kwargs)
        if num_columns:
            for i in range(num_columns):
                self.add_column()


def cwd():
    print(os.getcwd())


def filetime(path):
    mtime = os.path.getmtime(path)
    return str(datetime.datetime.fromtimestamp(mtime))[:-7]


def _validate_directory(directory):
    if not os.path.isdir(directory):
        console.print(f"[red]Directory {repr(directory)} not found")


def _print_if_match(table, pattern, root, s):
    if not pattern or re.search(pattern, s, flags=re.I):
        table.add_row(f"[blue][bold]{root}[/blue]/[yellow][bold]{s}", filetime(os.path.join(root, s)))
        return True
    else:
        return False


@interactive
def ff(directory: str = "C:\\", pattern: str = None):
    """Find files matching the given regex search"""
    _validate_directory(directory)
    table = ITable(show_header=False, num_columns=2)
    count = 0
    match_count = 0

    with Live(table, refresh_per_second=1):
        for root, folders, files in os.walk(directory):
            for file in files:
                count += 1
                if count % 50 == 0:
                    time.sleep(0.1)
                if _print_if_match(table, pattern, root, file):
                    match_count += 1

    print(f"Found {match_count} matches")



screen_w, screen_h = 2560, 1440
windows = []


def _enum_windows_callback(hwnd, _):
    global windows
    windows.append(hwnd)


def enum_windows():
    global windows
    windows = []
    win32gui.EnumWindows(_enum_windows_callback, None)
    return windows


def _find_window(pattern=None):
    enum_windows()
    window_map = {}
    for w in windows:
        text = win32gui.GetWindowText(w)
        if text:
            window_map[w] = text
    if pattern:
        for k, v in window_map.items():
            match = re.match(pattern, v)
            if match:
                return k
    else:
        window_names = list(window_map.values())
        window_names.sort()
        table = ITable(show_header=False, num_columns=1)
        for window_name in window_names:
            table.add_row(window_name)
        print(table)


def find_window(pattern):
    window = _find_window(pattern)
    if window:
        print(win32gui.GetWindowText(window))


MEDIA_PLAYER_PATTERN = re.compile("Filme & TV", re.I)
PYCHARM_PATTERN = re.compile("manimproject " + chr(8211) + " (.*)", re.I)


def _move_window(hwnd, x, y, w, h):
    win32gui.MoveWindow(hwnd, x, y, w, h, True)
    win32gui.SetForegroundWindow(hwnd)


def layout(num_windows=2):
    main_frame = win32ui.GetMainFrame().GetSafeHwnd()
    pycharm_window = _find_window(PYCHARM_PATTERN)
    media_player_window = _find_window(MEDIA_PLAYER_PATTERN)

    if num_windows > 1 and not pycharm_window:
        print("[red]PyCharm window not found")
        return
    if num_windows > 2 and not media_player_window:
        print("[red]Mediaplayer window not found")
        return

    w = 1160
    if num_windows == 1:
        _move_window(main_frame, 0, 0, screen_w, screen_h)
    elif num_windows == 2:
        _move_window(main_frame, screen_w - w, 0, w, screen_h - 20)
        _move_window(pycharm_window, 0, 0, screen_w - w, screen_h - 20)
    elif num_windows == 3:
        _move_window(main_frame, screen_w - w, (screen_h - 20) // 2, w, (screen_h - 20) // 2)
        _move_window(pycharm_window, 0, 0, screen_w - w, screen_h - 20)
        _move_window(media_player_window, screen_w - w, 0, w, (screen_h - 20) // 2)
    win32gui.SetForegroundWindow(main_frame)


@interactive
def scroll():
    while True:
        time.sleep(1.0)


def fd(directory: str = "C:\\", pattern: str = None):
    """Find directories matching the given regex search"""
    try:
        _validate_directory(directory)
        table = Table(show_header=False)
        table.add_column(width=100)
        table.add_column()
        for root, folders, files in os.walk(directory):
            for folder in folders:
                _print_if_match(table, pattern, root, folder)
        console.print(table)
    except KeyboardInterrupt:
        pass


SPLASH_TEXT = \
"""
8888ba.88ba                    oo                888888ba                    oo                     dP   
88  `8b  `8b                                     88    `8b                                          88   
88   88   88 .d8888b. 88d888b. dP 88d8b.d8b.    a88aaaa8P' 88d888b. .d8888b. dP .d8888b. .d8888b. d8888P 
88   88   88 88'  `88 88'  `88 88 88'`88'`88     88        88'  `88 88'  `88 88 88ooood8 88'  `""   88   
88   88   88 88.  .88 88    88 88 88  88  88     88        88       88.  .88 88 88.  ... 88.  ...   88   
dP   dP   dP `88888P8 dP    dP dP dP  dP  dP     dP        dP       `88888P' 88 `88888P' `88888P'   dP   
"""
