import os
import re
from collections import namedtuple, defaultdict
from dataclasses import dataclass

screen1_w, screen1_h = 1920, 1080
screen2_w, screen2_h = 2560, 1440
screen3_w, screen3_h = 1920, 1080


@dataclass(frozen=True)
class MonitorInfo:
    id: int
    pos: str
    width: int
    height: int


monitor_setup_map = defaultdict(lambda x: [MonitorInfo(2, "main", 1920, 1080)], {
    'LAPTOP-JTCKCJVQ': [MonitorInfo(2, "main", 2560, 1440)],
    'work': [MonitorInfo(1, "left", 1920, 1080),
             MonitorInfo(2, "main", 2560, 1440),
             MonitorInfo(3, "right", 1920, 1080)]
})

monitor_setup = monitor_setup_map[os.environ['COMPUTERNAME']]

IPYTHON_W = 1160  # 140 chars
MONITOR_INFO_MAIN = next(mi for mi in monitor_setup if mi.pos == "main")
try:
    MONITOR_INFO_RIGHT = next(mi for mi in monitor_setup if mi.pos == "right")
except StopIteration:
    MONITOR_INFO_RIGHT = None

if len(monitor_setup) == 1:
    ITOOLS_X = 0
    ITOOLS_WIDTH = MONITOR_INFO_MAIN.width
    ITOOLS_HEIGHT = MONITOR_INFO_MAIN.height
else:
    ITOOLS_X = MONITOR_INFO_MAIN.width
    ITOOLS_WIDTH = MONITOR_INFO_RIGHT.width
    ITOOLS_HEIGHT = MONITOR_INFO_RIGHT.height

IPYTHON_X_OFFSET = ITOOLS_WIDTH - IPYTHON_W
IPYTHON_H = ITOOLS_HEIGHT - 40
PYCHARM_POS = namedtuple('pos', 'x y')(x=ITOOLS_X, y=0)

MEDIA_PLAYER_PATTERN = re.compile("Filme & TV", re.I)
PYCHARM_TESTENV_PATTERN = re.compile("TestEnv " + chr(8211) + " (.*)", re.I)
PYCHARM_MANIMPROJECT_PATTERN = re.compile("manimproject " + chr(8211) + " (.*)", re.I)
PYCHARM_EXE = r"D:\Program Files\JetBrains\PyCharm Community Edition 2020.2.2\bin\pycharm64.exe"
NOTEPAD = r"C:\Program Files\Notepad++\notepad++.exe"
ECLIPSE_PATTERN = re.compile("eclipse-.*")
screen_w, screen_h = screen2_w, screen2_h

windowsconfig = {
    "1": {"main_frame": (ITOOLS_X + IPYTHON_X_OFFSET, 0, IPYTHON_W, MONITOR_INFO_MAIN.height)}
}
