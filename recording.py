import sys

sys.path.append("P:/Scripts")
sys.path.append("C:/DevTools/Python/Scripts")
sys.path.append("C:/Projekte/SSE/QS/TestEnv")
import win32api
# import setup_python_for_testenv as spt
# spt.uninstall("plumbum")
# spt.import_module("plumbum")
# spt.uninstall("opencv-python")
# spt.uninstall("pyautogui")
# spt.uninstall("numpy")
# spt.uninstall("rpyc")
# spt.import_module("cv2", package_name="opencv-python")
# spt.import_module("pyautogui")
# spt.import_module("numpy")
# spt.import_module("rpyc")
import rpyc
import cv2 as cv  # OpenCV
import pyautogui
import numpy as np
import timeit
import time
import os
import re
from threading import Thread
from typing import Sequence, Tuple
from collections import namedtuple
import datetime

fps = 6
stop_recording = False
SERVER_PORT = 12345


def stop():
    global stop_recording
    stop_recording = True


_DEFAULT_VIDEO_CAPTURE_NAME = "Recorded.wmv"


def get_desktop_size():
    return win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)


def record(output_path=_DEFAULT_VIDEO_CAPTURE_NAME, *, show_window=False, capture=False, max_duration=600):
    global stop_recording
    scripts_path = os.path.join(os.environ['USERPROFILE'], r"AppData\Roaming\Python\Python38\Scripts")

    if scripts_path not in sys.path:
        sys.path.append(scripts_path)

    from rpyc_classic import ClassicServer
    from rpyc.utils.server import ThreadedServer
    from rpyc import SlaveService

    server = ThreadedServer(SlaveService, port=SERVER_PORT, ipv6=False)
    print(f"recording to {output_path} (port: {server.port})")
    server.logger.quiet = True
    thread = Thread(target=server.start)
    thread.start()

    codec = cv.VideoWriter_fourcc(*"WMV2")
    w, h = get_desktop_size()
    out = cv.VideoWriter(output_path, codec, fps, (w, h))

    if show_window:
        cv.namedWindow("Recording", cv.WINDOW_NORMAL)
        cv.resizeWindow("Recording", w, h)
    if capture:
        cap = cv.VideoCapture(0)
    frame_count = 0
    last_time = timeit.default_timer()
    idle_count = 0
    start_time = last_time
    while not max_duration or timeit.default_timer() - start_time < max_duration:
        if capture:
            ret, frame = cap.read()
            assert ret
            frame = cv.flip(frame, 0)
        else:
            try:
                img = pyautogui.screenshot()
            except Exception as ex:
                print("screenshot failed")
                continue
            frame = np.array(img)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        out.write(frame)
        if show_window:
            cv.imshow('Recording', frame)
        now = timeit.default_timer()
        # expected total time: frame_count * fps
        # actual total time: now - start_time
        delay = frame_count / fps - now + start_time
        if delay > 0:
            time.sleep(delay)
            idle_count += 1
        frame_count += 1

        if show_window:
            if cv.waitKey(1) == ord('q'):
                break

        if stop_recording:
            print("record - stop signal")
            stop_recording = False
            break

    duration = timeit.default_timer() - start_time
    print(f"wrote {frame_count} frames ({duration:.3f} sec, {frame_count / duration} fps, idle: {idle_count})")
    if capture:
        cap.release()

    out.release()  # closing the video file
    if show_window:
        cv.destroyAllWindows()  # destroying the recording window

    server.close()
    thread.join()
    print("record - server closed")


def _substitute_umlaute(s):
    umlaute, subs = "äöüÄÖÜß", ["ae", "oe", "ue", "Ae", "Oe", "Ue", "ss"]
    for i, umlaut in enumerate(umlaute):
        s = s.replace(umlaut, subs[i])
    return s


def play(filepath=_DEFAULT_VIDEO_CAPTURE_NAME,
         timeout=20.0,
         repeat=True,
         captions=Sequence[Tuple[float, str, str, str]],
         offset=3.5,
         pause=2,
         show_window=False):
    if show_window:
        from itools import _set_foreground_window, _find_window
        window = cv.namedWindow(filepath, cv.WINDOW_NORMAL)
        w, h = get_desktop_size()
        cv.resizeWindow(filepath, w // 2, h // 2)
        time.sleep(0.01)
        hwnd = _find_window(filepath)
        if hwnd:
            _set_foreground_window(hwnd)

    try:
        cap = cv.VideoCapture(filepath)

        if not show_window:
            w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            output_path = filepath.replace(".wmv", "-render.wmv")
            out = cv.VideoWriter(output_path, int(cap.get(cv.CAP_PROP_FOURCC)), cap.get(cv.CAP_PROP_FPS), (w, h))
            if not out.isOpened():
                print(f"can't access file {output_path}")
                cap.release()
                out.release()
                return

        Pos = namedtuple("pos", "x y")

        pos = Pos(x=60, y=120)
        font = cv.FONT_HERSHEY_COMPLEX
        fontscale = 1
        RECT_MARGIN = 12
        start_time = timeit.default_timer()
        it = iter(captions)
        try:
            next_caption = next(it)
        except StopIteration:
            next_caption = None

        frame_count = 0
        step = None
        idle_count = 0
        old_step = None
        old_scenario = None
        old_feature = None

        def text(s, pos, color=(0xff, 0xff, 0xff)):
            textsize = cv.getTextSize(s, font, fontscale, 2)[0]
            textwidth, textheight = tuple(textsize)
            textwidth2 = textwidth + 4
            textheight2 = cv.getTextSize(s, font, fontscale * textwidth2 / textwidth, 2)[0][1]
            subimg_coords = pos.y - textheight2 - RECT_MARGIN, pos.y + textheight2 - textheight + RECT_MARGIN,\
                            pos.x, pos.x + textwidth2
            subimg_h, subimg_w = subimg_coords[1] - subimg_coords[0], subimg_coords[3] - subimg_coords[2]
            subimg = frame[subimg_coords[0]:subimg_coords[1], subimg_coords[2]:subimg_coords[3]]
            if subimg.shape[1] < subimg_w or subimg.shape[0] < subimg_h:
                subimg_w = min(subimg_w, subimg.shape[1])
                subimg_h = min(subimg_h, subimg.shape[0])
            new_subimg = np.zeros((subimg_h, subimg_w, 3), np.uint8)
            try:
                new_subimg = cv.addWeighted(new_subimg, 1.0, subimg, 0.7, 1.0)
            except cv.error as ex:
                print(ex)
            # assert new_subimg is not None, f"new_subimg: {new_subimg}"
            if new_subimg is not None:
                frame[subimg_coords[0]:subimg_coords[0] + subimg_h, subimg_coords[2]:subimg_coords[2] + subimg_w] = new_subimg
                cv.putText(frame, s, (pos.x - 3, pos.y), font, fontscale * textwidth2 / textwidth, (120, 80, 80),
                           5, cv.LINE_AA)
                cv.putText(frame, s, (pos.x, pos.y), font, fontscale, color, 2, cv.LINE_AA)
            else:
                print(f"error creating subimage at frame {frame_count}")
            return Pos(x=pos.x, y=pos.y + textheight + 18)

        while cap.isOpened() and (not timeout or timeit.default_timer() - start_time < timeout):
            ret, frame = cap.read()
            if frame is not None:
                if next_caption and next_caption[0] <= frame_count / fps + offset:
                    step, scenario, feature = tuple(next_caption[1:])
                    step = _substitute_umlaute(step)
                    scenario = _substitute_umlaute(scenario)
                    feature = _substitute_umlaute(feature)

                    if step != old_step or scenario != old_scenario or feature != old_feature:
                        try:
                            print(f"new caption at {frame_count / fps}")
                            next_caption = next(it)
                        except StopIteration:
                            pass
                if step:
                    # text ="ä"
                    # cv.addText(frame, text, (pos.x, pos.y), "Times")

                    if (step != old_step or scenario != old_scenario or feature != old_feature) and pause:
                        frame = cv.blur(frame, (5, 5))
                        print("blurred pause")

                    pos = text(feature, Pos(60, 40), (0x50, 0xb0, 0x50))
                    pos = text(scenario, pos, (0x80, 0xc0, 0x55))
                    pos = text(step, pos, (0xff, 0xff, 0xff))

                if show_window:
                    key = cv.waitKey(1) & 0xFF
                    if key != 0xFF:
                        print(f"playback aborted (key {repr(chr(key))})")
                        break
                    cv.imshow(filepath, frame)

                    # expected total time: frame_count * fps
                    # actual total time: now - start_time
                    now = timeit.default_timer()
                    delay = frame_count / fps - now + start_time
                    if delay > 0:
                        time.sleep(delay)
                        idle_count += 1
                else:
                    out.write(frame)

                    if pause and (step != old_step or scenario != old_scenario or feature != old_feature):
                        for i in range(int(pause * fps)):
                            out.write(frame)
                        old_step = step
                        old_scenario = scenario
                        old_feature = feature

                frame_count += 1
            else:
                if repeat:
                    cap.release()
                    cap = cv.VideoCapture(filepath)
                else:
                    break

    finally:
        time.sleep(0.001)
        cap.release()
        if show_window:
            cv.destroyAllWindows()
        else:
            out.release()  # closing the video file


if __name__ == '__main__':
    # record("movie.wmv", max_duration=20)
    # logpath = "T:/Tests/out/SSE_Win10/run_Win10_minimal.log"
    # movie = "C:/Projekte/SSE/QS/Tests/features/Behave-Inhaltlich_1_Gewinnermittlung_tr238_ELSTER-Versand.wmv"
    # movie = "T:/Tests/features/Behave-Minimal.wmv"

    # logpath = "T:/Tests/out/SSE_Win10/run_Win10_stable_3.log"
    # movie = "T:/Tests/features/Behave-Technisch_3.wmv"

    logpath = "T:/Tests/out/SSE_Win10/run_Win10_inhaltlich_stable_1.log"
    movie = "T:/Tests/features/Behave-Inhaltlich_1.wmv"

    # logpath = "C:/Projekte/SSE/QS/Tests/out/run.log"
    with open(logpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    re_timestamp = re.compile(r"(\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d) .*")
    re_step = re.compile(r".* - BEFORE STEP: (.*) # .*")
    re_feature = re.compile(r".* - BEFORE FEATURE: (.*) # .*")
    re_scenario = re.compile(r".* - BEFORE SCENARIO: (.*) # .*")
    steps = []
    start_time = None
    first_step_time = None
    feature = None
    scenario = None
    for line in lines:
        match_feature = re.match(re_feature, line)
        if match_feature:
            feature = match_feature[1]
        else:
            match_scenario = re.match(re_scenario, line)
            if match_scenario:
                scenario = match_scenario[1]
            else:
                match_step = re.match(re_step, line)
                if match_step:
                    step = match_step[1]
                    match_timestamp = re.match(re_timestamp, line)
                    if match_timestamp:
                        timestamp = datetime.datetime.strptime(match_timestamp[1], '%Y-%m-%d %H:%M:%S,%f')
                        steps.append((timestamp, step, scenario, feature))
                        if first_step_time is None:
                            first_step_time = timestamp
                elif line.endswith(" - start recording\n"):
                    match_timestamp = re.match(re_timestamp, line)
                    start_time = datetime.datetime.strptime(match_timestamp[1], '%Y-%m-%d %H:%M:%S,%f')
    # assert first_step_time < start_time, f"first_step_time: {first_step_time}, start_time: {start_time}"
    if first_step_time < start_time:
        time_offset = start_time - first_step_time
    else:
        time_offset = datetime.timedelta(0)

    for timestamp, step, scenario, feature in steps:
        if start_time:
            print(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S,%f')}: {step} {timestamp - start_time}s")
    captions = [((ts - first_step_time).total_seconds(), step, scenario, feature) for ts, step, scenario, feature in
                steps]
    # play("T:/Tests/Behave.wmv", timeout=None, captions=captions)
    print(f"{len(captions)} captions")
    play(movie, timeout=0, captions=captions, repeat=False, show_window=False, pause=2,
         offset=(time_offset + datetime.timedelta(seconds=6)).total_seconds())
