import sys
import win32api
import cv2 as cv  # OpenCV
import pyautogui
import numpy as np
import timeit
import time
import os
from threading import Thread
from collections import namedtuple
from typing import Sequence

stop_recording = False

SERVER_PORT = 0


def stop():
    global stop_recording
    stop_recording = True


_DEFAULT_VIDEO_CAPTURE_NAME = "Recorded.wmv"


def get_desktop_size():
    return win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)


def record(output_path=_DEFAULT_VIDEO_CAPTURE_NAME, *, show_window=False, capture=False, max_duration=None):
    global stop_recording
    scripts_path = os.path.join(os.environ['USERPROFILE'], r"AppData\Roaming\Python\Python38\Scripts")

    if scripts_path not in sys.path:
        sys.path.append(scripts_path)

    from rpyc_classic import ClassicServer
    from rpyc.utils.server import ThreadedServer
    from rpyc import SlaveService

    server = ThreadedServer(SlaveService, port=SERVER_PORT, ipv6=True)
    print(f"port: {server.port}")
    server.logger.quiet = True
    thread = Thread(target=server.start)
    thread.start()

    codec = cv.VideoWriter_fourcc(*"WMV2")
    fps = 12
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
            img = pyautogui.screenshot()
            frame = np.array(img)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        out.write(frame)
        if show_window:
            cv.imshow('Recording', frame)
        while timeit.default_timer() - last_time < 1 / fps:
            idle_count += 1
            time.sleep(0.001)
        last_time = timeit.default_timer()
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


def play(filepath=_DEFAULT_VIDEO_CAPTURE_NAME, timeout=20.0, repeat=True, captions=Sequence[tuple[float][str]]):
    from itools import _set_foreground_window, _find_window
    starttime = timeit.default_timer()
    window = cv.namedWindow(filepath, cv.WINDOW_NORMAL)
    w, h = get_desktop_size()
    cv.resizeWindow(filepath, w // 2, h // 2)
    time.sleep(0.01)
    hwnd = _find_window(filepath)
    if hwnd:
        _set_foreground_window(hwnd)

    try:
        cap = cv.VideoCapture(filepath)
        pos = namedtuple("pos", "x y")(x=60, y=100)
        text = "0123456789" * 4
        font = cv.FONT_HERSHEY_COMPLEX
        fontscale = 2
        RECT_MARGIN = 12

        while cap.isOpened() and (not timeout or timeit.default_timer() - starttime < timeout):
            ret, frame = cap.read()
            if frame is not None:
                textsize = cv.getTextSize(text, font, fontscale, 2)[0]
                textwidth, textheight = tuple(textsize)
                textwidth2 = textwidth + 4
                textheight2 = cv.getTextSize(text, font, fontscale * textwidth2 / textwidth, 2)[0][1]
                subimg_coords = pos.y - textheight2 - RECT_MARGIN, pos.y + textheight2 - textheight + RECT_MARGIN, pos.x, pos.x + textwidth2
                subimg_h, subimg_w = subimg_coords[1] - subimg_coords[0], subimg_coords[3] - subimg_coords[2]
                subimg = frame[subimg_coords[0]:subimg_coords[1], subimg_coords[2]:subimg_coords[3]]
                new_subimg = np.zeros((subimg_h, subimg_w, 3), np.uint8)
                new_subimg = cv.addWeighted(new_subimg, 1.0, subimg, 0.7, 1.0)
                frame[subimg_coords[0]:subimg_coords[1], subimg_coords[2]:subimg_coords[3]] = new_subimg
                cv.putText(frame, text, (pos.x - 3, pos.y), font, fontscale * textwidth2 / textwidth, (120, 80, 80), 5, cv.LINE_AA)
                cv.putText(frame, text, (pos.x, pos.y), font, fontscale, (255, 255, 255), 2, cv.LINE_AA)
                cv.imshow(filepath, frame)
                key = cv.waitKey(1) & 0xFF
                if key != 0xFF:
                    print(f"playback aborted (key {repr(chr(key))})")
                    break
                time.sleep(0.001)
            else:
                if repeat:
                    cap.release()
                    cap = cv.VideoCapture(filepath)
                else:
                    break
    finally:
        time.sleep(0.001)
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    pass
    # record()
