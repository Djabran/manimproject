import subprocess
import os
import re
import datetime
import time
from rich.table import Table
from rich import print
import os, time
import re
from rich import print
from rich.live import Live
import datetime
from rich.console import Console
from rich.prompt import Prompt
import win32ui
import win32gui
from context import *
console = Console(width=200)

screen2_w, screen2_h = 2560, 1440
screen3_w = 1920
IPYTHON_W = 1160
IPYTHON_H = screen2_h - 40
PYCHARM_POS = (screen3_w, 0)
IPYTHON_X = screen3_w + screen2_w - IPYTHON_W
MEDIA_PLAYER_PATTERN = re.compile("Filme & TV", re.I)
PYCHARM_TESTENV_PATTERN = re.compile("TestEnv " + chr(8211) + " (.*)", re.I)
ECLIPSE_PATTERN = re.compile("eclipse-.*")
windows = []


def _enum_windows_callback(hwnd, _):
    global windows
    windows.append(hwnd)


def enum_windows():
    global windows
    windows = []
    win32gui.EnumWindows(_enum_windows_callback, None)
    return windows


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


def interactive(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except KeyboardInterrupt:
            pass
    return wrapper


class ITable(Table):
    def __init__(self, *args, num_columns=None, **kwargs):
        super().__init__(*args, **kwargs)
        if num_columns:
            for i in range(num_columns):
                self.add_column()


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


def date(timestamp) -> str:
    """
    File timstamp as readable date string
    """
    return datetime.datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S')


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


def _walk(walkfunc, directory: str = "C:\\", pattern: str = None):
    _validate_directory(directory)
    table = ITable(show_header=False, num_columns=2)
    match_count = 0

    with Live(table, refresh_per_second=1):
        for root, folders, files in os.walk(directory):
            matchcount = walkfunc(root, folders, files, pattern)

    print(f"Found {match_count} matches")


@interactive
def ff(directory: str = "C:\\", pattern: str = None):
    """Find files matching the given regex search"""

    def _walkfunc(root, folders, files, pattern):
        count = 0
        match_count = 0
        for file in files:
            count += 1
            if count % 50 == 0:
                time.sleep(0.1)
            if _print_if_match(table, pattern, root, file):
                match_count += 1
        return match_count

    _walk(directory, pattern, _walkfunc)


@interactive
def fd(directory: str = "C:\\", pattern: str = None):
    """Find directories matching the given regex search"""

    def _walkfunc(root, folders, files, pattern):
        count = 0
        match_count = 0
        for folder in folders:
            count += 1
            if count % 50 == 0:
                time.sleep(0.1)
            if _print_if_match(table, pattern, root, folder):
                match_count += 1
        return match_count

    _walk(directory, pattern, _walkfunc)


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


def _move_window(hwnd, x, y, w, h):
    win32gui.MoveWindow(hwnd, x, y, w, h, True)
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(0.1)


def layout(num_windows=2):
    main_frame = win32ui.GetMainFrame().GetSafeHwnd()
    pycharm_testenv_window = _find_window(PYCHARM_TESTENV_PATTERN )
    media_player_window = _find_window(MEDIA_PLAYER_PATTERN)
    eclipse_window = _find_window(ECLIPSE_PATTERN)

    if num_windows > 1 and not pycharm_testenv_window :
        print("[red]PyCharm window not found")
        os.system('"' + PYCHARM + '" C:/Projekte/SSE/QS/TestEnv')
        return
    if int(num_windows) > 2 and not media_player_window:
        print("[red]Mediaplayer window not found")
        return

    if num_windows == 1:
        _move_window(main_frame, IPYTHON_X, 0, screen2_w, screen2_h)
    elif int(num_windows) == 2:
        _move_window(main_frame, IPYTHON_X, 0, IPYTHON_W, IPYTHON_H)
        _move_window(pycharm_testenv_window , PYCHARM_POS[0], PYCHARM_POS[1], screen2_w - IPYTHON_W, IPYTHON_H)
        if num_windows == 2.1:
            _move_window(eclipse_window, PYCHARM_POS[0], PYCHARM_POS[1], screen2_w - IPYTHON_W, IPYTHON_H)
    elif num_windows == 3:
        _move_window(main_frame, IPYTHON_X, (screen2_h - 20) // 2, IPYTHON_W, IPYTHON_H // 2)
        _move_window(pycharm_testenv_window , 0, 0, screen2_w - w, screen2_h - 20)
        _move_window(media_player_window, screen2_w - IPYTHON_W, 0, IPYTHON_W, IPYTHON_H // 2)
    win32gui.SetForegroundWindow(main_frame)


@interactive
def scroll():
    while True:
        time.sleep(10.0)
