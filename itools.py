import sys
import json
import datetime
import os
import re
import shutil
import time
import timeit
import subprocess
import pywintypes
import win32gui
import win32ui
import appdirs
from collections import defaultdict
from rich import print
from rich.console import Console
from rich.live import Live
from rich.table import Table
from windowsconfig import *
from typing import Union, Iterable

windows = []

CODING_MAP = defaultdict(lambda: 'ansi',
                         {
                             ".py": "utf-8",
                             ".feature": "utf-8",
                         })

TABLE_UPDATE_COUNT = 1000

console = Console(width=140)


def interactive(func):
    def wrapper(*args, **kwargs):
        """
        wrapper for all commands that can be interrupted with Ctrl-C
        """
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("[cyan]aborted.")

    return wrapper


class ITable(Table):
    def __init__(self, *args, num_columns=None, **kwargs):
        super().__init__(*args, **kwargs)
        if num_columns:
            for i in range(num_columns):
                self.add_column()


def _collect_if_match(collection, live, pattern, root, s):
    if not pattern or re.search(pattern, s, flags=re.I):
        collection.append((root, s, os.path.getmtime(os.path.join(root, s))))
        return True
    else:
        return False


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
    _set_foreground_window(hwnd)


def _set_foreground_window(hwnd):
    try:
        win32gui.SetForegroundWindow(hwnd)
    except pywintypes.error as err:
        if err.winerror:
            raise err
        else:
            time.sleep(0.2)
            win32gui.SetForegroundWindow(hwnd)


def _table_from_collection(directory, root, collection):
    table = Table()
    table.add_column(f"[blue]{os.path.normpath(root)}[/blue]")
    table.add_column()
    collection.sort(key=lambda x: x[2])
    for folder, filename, timestamp in collection:
        if folder.startswith(directory):
            folder = folder[len(directory):]
        folder_str = folder + '\\\\' if folder else ""
        if folder_str.startswith('\\') or folder_str.startswith('/'):
            folder_str = folder_str[1:]
        path = f"[bold][blue]{folder_str}[bold][yellow]{filename}"
        table.add_row(path, date(timestamp))
    return table


def _tail_f(filepath, encoding='cp1252'):
    first_call = True
    while True:
        try:
            with open(filepath) as f:
                latest_data = f.read()
                while True:
                    if '\n' not in latest_data:
                        latest_data += f.read()
                        if '\n' not in latest_data:
                            yield None
                            if not os.path.isfile(filepath):
                                break
                            continue
                    latest_lines = latest_data.split('\n')
                    if latest_data[-1] != '\n':
                        latest_data = latest_lines[-1]
                    else:
                        latest_data = f.read()
                    for line in latest_lines[:-1]:
                        yield line + '\n'
        except IOError:
            yield 'io error'


def _validate_directory(directory):
    if not os.path.isdir(directory):
        console.print(f"[red]Directory {repr(directory)} not found")


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
            count, steps = walkfunc(collection, live, root, folders, files, steps)
            match_count += count

        if len(collection) % TABLE_UPDATE_COUNT:
            live.update(_table_from_collection(directory, directory, collection))
        else:
            if live.get_renderable().columns[0].header != directory:
                live.update(_table_from_collection(directory, directory, collection))
    print(f"Found {match_count} matches")


def cwd():
    print(os.getcwd())


def date(timestamp) -> str:
    """
    File timstamp as readable date string
    """
    return datetime.datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S')


@interactive
def fd(directory: str = "C:\\", pattern: str = None):
    """ Find directories matching the given regex search """

    def walkfunc(collection, live, root, folders, files, steps):
        count = 0
        match_count = 0
        for folder in folders:
            count += 1
            if _collect_if_match(collection, live, pattern, root, folder):
                match_count += 1
            steps += 1
            if not steps % TABLE_UPDATE_COUNT:
                live.update(_table_from_collection(directory, root, collection))
                time.sleep(0.05)
        return match_count, steps

    _walk(walkfunc, directory)


@interactive
def ff(directory: str = ".", pattern: str = None):
    """ Find files matching the given regex search """

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
    Regex search in files contents (see CODING_MAP)
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
    pycharm_manimproject_window = _find_window(PYCHARM_MANIMPROJECT_PATTERN)
    pycharm_testenv_window = _find_window(PYCHARM_TESTENV_PATTERN)
    media_player_window = _find_window(MEDIA_PLAYER_PATTERN)
    eclipse_window = _find_window(ECLIPSE_PATTERN)

    if num_windows > 1 and not pycharm_testenv_window:
        print("[red]PyCharm window not found")
        return

    w = 1160
    if num_windows == 1:
        _move_window(main_frame, IPYTHON_X, 0, screen2_w, screen2_h)
    elif int(num_windows) == 2:
        _move_window(main_frame, IPYTHON_X, 0, IPYTHON_W, IPYTHON_H)
        _move_window(pycharm_testenv_window, PYCHARM_POS[0], PYCHARM_POS[1], screen2_w - IPYTHON_W, IPYTHON_H)
        if num_windows == 2.1:
            _move_window(eclipse_window, PYCHARM_POS[0], PYCHARM_POS[1], screen2_w - IPYTHON_W, IPYTHON_H)
    elif num_windows == 3:
        _move_window(main_frame, IPYTHON_X, (screen2_h - 20) // 2, IPYTHON_W, IPYTHON_H // 2)
        _move_window(pycharm_testenv_window, 0, 0, screen2_w - w, screen2_h - 20)
        _move_window(media_player_window, screen2_w - IPYTHON_W, 0, IPYTHON_W, IPYTHON_H // 2)
    _set_foreground_window(main_frame)


def rm(path, pattern=None):
    if not os.path.exists(path):
        print(f"Path {path} doesn't exist")
        return
    if pattern:
        if os.path.isdir(path):
            directory = path
            def walkfunc(collection, live, root, folders, files, steps):
                count = 0
                match_count = 0
                for file in files:
                    count += 1
                    if re.search(pattern, file):
                        print(f"deleting {root}{file}")
                        os.unlink(os.path.join(root, file))
                return match_count, steps

            _walk(walkfunc, directory)
        else:
            print(f"{path} is not a directory")
    else:
        if os.path.isdir(path):
            shutil.rmtree(path)


_COPY_STATS_PATH = os.path.join(appdirs.user_data_dir(), "copy_stats.json")
_COPY_STATS_MAXLEN = 42
copy_stats = defaultdict(lambda: [])

try:
    with open(_COPY_STATS_PATH, "r", encoding='utf-8') as f:
        copy_stats = defaultdict(lambda: [], json.load(f))
except FileNotFoundError:
    print(f"{_COPY_STATS_PATH} not found")
except Exception as ex:
    console.print_exception(show_locals=True)
    copy_stats = defaultdict(lambda: [])


@interactive
def robocopy(*args, **kwargs):
    _robocopy(*args, **kwargs)


def _robocopy(src, dst, pattern: Iterable = None, exclude_dir: Iterable = None, purge=True):
    # print(f"robocopy {src} -> {dst} pattern: {pattern} exclude_dir: {exclude_dir}...")
    key = src + "->" + dst
    if key in copy_stats:
        est = copy_stats[key]
        avg = sum(est) / len(est)
        print(f"Average time: {avg:0.3f} secs")
    start_time = timeit.default_timer()

    args = ["robocopy", src, dst]
    if pattern:
        args += pattern.split() if type(pattern) is str else pattern
    args += ["/S"]
    if purge:
        args += ["/PURGE"]
    if exclude_dir:
        args += ["/XD", exclude_dir] if type(exclude_dir) is str else ["/XD", *exclude_dir]
    print(f"robocopy {args}...")
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    dur = timeit.default_timer() - start_time
    copy_stats[key].append(dur)
    if len(copy_stats[key]) > _COPY_STATS_MAXLEN:
        copy_stats[key] = copy_stats[key][len(copy_stats[key]) - _COPY_STATS_MAXLEN:]

    with open(_COPY_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(copy_stats, f)

    s = stdout.decode('cp850')
    if s:
        print("----- stdout -------")
        print(s)
    s = stderr.decode('cp850')
    if s:
        print("[red]----- stderr -------")
        print("[red]" + s)
    return proc.returncode


@interactive
def scroll():
    while True:
        time.sleep(1.0)


@interactive
def tail(filepath, encoding='ansi'):
    tail_file = _tail_f(filepath, encoding)
    while True:
        try:
            s = next(tail_file)
            if s:
                sys.stdout.write(s)
        except StopIteration:
            break
