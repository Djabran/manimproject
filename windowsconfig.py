import re

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

windowsconfig = {
    "1": {
        "main_frame" : (IPYTHON_X, 0, screen2_w, screen2_h)
}}