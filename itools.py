import datetime
import os
import re
import shutil
import time

import pywintypes
import win32gui
import win32ui
from rich import print
from rich.console import Console
from rich.live import Live
from rich.table import Table

screen2_w, screen2_h = 2560, 1440
screen3_w = 1920
IPYTHON_W = 1160
IPYTHON_H = screen2_h - 40
PYCHARM_POS = (screen3_w, 0)
IPYTHON_X = screen3_w + screen2_w - IPYTHON_W
MEDIA_PLAYER_PATTERN = re.compile("Filme & TV", re.I)
PYCHARM_TESTENV_PATTERN = re.compile("TestEnv " + chr(8211) + " (.*)", re.I)
PYCHARM_MANIMPROJECT_PATTERN = re.compile("manimproject " + chr(8211) + " (.*)", re.I)
PYCHARM_EXE = r"D:\Program Files\JetBrains\PyCharm Community Edition 2020.2.2\bin\pycharm64.exe"
NOTEPAD = r"C:\Program Files\Notepad++\notepad++.exe"

ECLIPSE_PATTERN = re.compile("eclipse-.*")
screen_w, screen_h = screen2_w, screen2_h
windows = []

CODING_MAP = {
    ".py": "utf-8",
    ".bat": "cp1252",
    ".ddf": "ansi",
    ".inc": "ansi",
    ".adl": "ansi",
    ".yaddl": "ansi",
    ".template": "ansi",
    ".log": "ansi",
    ".feature": "utf-8",
    ".edf": "ansi"
}

console = Console(width=200)


def interactive(func):
    def wrapper(*args):
        try:
            func(*args)
        except KeyboardInterrupt:
            print("[cyan]aborted.")

    return wrapper


class ITable(Table):
    def __init__(self, *args, num_columns=None, **kwargs):
        super().__init__(*args, **kwargs)
        if num_columns:
            for i in range(num_columns):
                self.add_column()


def _find_window(pattern=None):
    _enum_windows()
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


def _enum_windows_callback(hwnd, _):
    global windows
    windows.append(hwnd)


def _enum_windows():
    global windows
    windows = []
    win32gui.EnumWindows(_enum_windows_callback, None)
    return windows


def _move_window(hwnd, x, y, w, h):
    win32gui.MoveWindow(hwnd, x, y, w, h, True)
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(0.1)


def _validate_directory(directory):
    if not os.path.isdir(directory):
        console.print(f"[red]Directory {repr(directory)} not found")


def cwd():
    print(os.getcwd())


def date(timestamp) -> str:
    """
    File timstamp as readable date string
    """
    return datetime.datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S')


TABLE_UPDATE_COUNT = 1000


def _table_from_collection(directory, root, collection):
    table = Table()
    table.add_column(f"[blue]{root}[/blue]")
    table.add_column()
    collection.sort(key=lambda x: x[2])
    for folder, filename, timestamp in collection:
        # path = os.path.join(folder, filename)
        if folder.startswith(directory):
            folder = folder[len(directory) + 1:]
        folder_str = folder + '\\\\' if folder else ""
        path = f"[bold][blue]{folder_str}[bold][yellow]{filename}"
        table.add_row(path, date(timestamp))
    return table


def _walk(walkfunc, directory: str = "C:\\"):
    _validate_directory(directory)
    table = ITable(show_header=False, num_columns=2)
    match_count = 0
    collection = []
    steps = 0
    with Live(table, refresh_per_second=0.5) as live:
        time.sleep(0.4)
        content = os.walk(directory)
        for root, folders, files in content:
            match_count, steps = walkfunc(collection, live, root, folders, files, steps)

        if len(collection) % TABLE_UPDATE_COUNT:
            live.update(_table_from_collection(directory, directory, collection))
        else:
            if live.get_renderable().columns[0].header != directory:
                live.update(_table_from_collection(directory, directory, collection))

    print(f"Found {match_count} matches")


def _collect_if_match(collection, live, pattern, root, s):
    if not pattern or re.search(pattern, s, flags=re.I):
        collection.append((root, s, os.path.getmtime(os.path.join(root, s))))
        return True
    else:
        return False


@interactive
def fd(directory: str = "C:\\", pattern: str = None):
    """Find directories matching the given regex search"""

    def _walkfunc(collection, live, root, folders, files):
        count = 0
        match_count = 0
        for folder in folders:
            count += 1
            if count % 50 == 0:
                time.sleep(0.1)
            if _collect_if_match(collection, live, pattern, root, folder):
                match_count += 1
        return match_count

    _walk(_walkfunc, directory)


@interactive
def ff(directory: str = "C:\\", pattern: str = None):
    """Find files matching the given regex search"""

    def walkfunc(collection, live, root, folders, files, steps):
        count = 0
        match_count = 0
        for file in files:
            count += 1
            if _collect_if_match(collection, live, pattern, root, file):
                match_count += 1
            steps += 1
            if not steps % TABLE_UPDATE_COUNT:
                live.update(_table_from_collection(directory, root, collection))
                time.sleep(0.05)
        return match_count, steps

    _walk(walkfunc, directory)


@interactive
def fif(directory, regex=".*", case_insensitive=True):
    """
    Find in files
    """

    def _search_file(encoding):
        with open(filepath, "rb") as f:
            buffer = f.read()
            try:
                s = buffer.decode(encoding)
            except UnicodeDecodeError as ex:
                print(f"{filepath}: {ex}")
            else:
                lines = s.splitlines(keepends=False)
                found = False
                for i, line in enumerate(lines):
                    if re.search(regex, line, flags=re.I if case_insensitive else 0):
                        found = True
                        print(f"{filepath}:{i + 1}: {line}")
                return found

    print(f"find {repr(regex)} in {directory}{' (case insensitive)' if case_insensitive else ''}...")
    all_files = []
    found = False
    for root, folder, files in os.walk(directory, topdown=True):
        files = [f for f in files if os.path.isfile(root + "/" + f)]
        for filepath in files:
            filepath = root + "/" + filepath
            name, ext = os.path.splitext(filepath)
            if ext in CODING_MAP:
                all_files.append(filepath)
                found |= _search_file(CODING_MAP[ext])
    if not found:
        yn = Prompt.ask(f"Nothing found, show Files? ", default="n", choices=["y", "n"])
        if yn == "y":
            for f in all_files:
                print(f)


def filetime(path):
    mtime = os.path.getmtime(path)
    return str(datetime.datetime.fromtimestamp(mtime))[:-7]


def find_window(pattern):
    window = _find_window(pattern)
    if window:
        print(win32gui.GetWindowText(window))


def layout(num_windows=2):
    main_frame = win32ui.GetMainFrame().GetSafeHwnd()
    pycharm_window = _find_window(PYCHARM_MANIMPROJECT_PATTERN)
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
    set_foreground_window(main_frame)


def rm(path):
    if os.path.isdir(path):
        shutil.rmtree(path)


@interactive
def robocopy(src, dst, pattern: str = None, exclude_dir: str = None, purge=True):
    print(f"robocopy {src} -> {dst} pattern: {pattern} exclude_dir: {exclude_dir}...")

    args = ["robocopy", src, dst]
    if pattern:
        args += [pattern]
    args += ["/S"]
    if purge:
        args += ["/PURGE"]
    if exclude_dir:
        args += ["/XD", exclude_dir]

    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    print("----- stdout -------")
    print(stdout.decode('cp850'))
    print("[red]----- stderr -------")
    print("[red]" + stderr.decode('cp850'))


@interactive
def scroll():
    while True:
        time.sleep(1.0)


def set_foreground_window(hwnd):
    try:
        win32gui.SetForegroundWindow(hwnd)
    except pywintypes.error as err:
        if err.winerror:
            raise err
        else:
            time.sleep(0.2)
            win32gui.SetForegroundWindow(hwnd)


SPLASH_TEXT = \
    """
    8888ba.88ba                    oo                888888ba                    oo                     dP   
    88  `8b  `8b                                     88    `8b                                          88   
    88   88   88 .d8888b. 88d888b. dP 88d8b.d8b.    a88aaaa8P' 88d888b. .d8888b. dP .d8888b. .d8888b. d8888P 
    88   88   88 88'  `88 88'  `88 88 88'`88'`88     88        88'  `88 88'  `88 88 88ooood8 88'  `""   88   
    88   88   88 88.  .88 88    88 88 88  88  88     88        88       88.  .88 88 88.  ... 88.  ...   88   
    dP   dP   dP `88888P8 dP    dP dP dP  dP  dP     dP        dP       `88888P' 88 `88888P' `88888P'   dP   
    """
