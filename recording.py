import sys
sys.path.append("P:/Scripts")
sys.path.append("C:/DevTools/Python/Scripts")
sys.path.append("C:/Projekte/SSE/QS/TestEnv")
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
import cv2 as cv # OpenCV
import pyautogui
import numpy as np
import timeit
import time
import os
import threading

stop_recording = False

SERVER_PORT = 0

def stop():
    global stop_recording
    stop_recording = True


def record(output_path = "Recorded.wmv", *, show_window=False, capture=False, max_duration=None):
    global stop_recording
    scripts_path = os.path.join(os.environ['USERPROFILE'], r"AppData\Roaming\Python\Python38\Scripts")
    
    if scripts_path not in sys.path:
        sys.path.append(scripts_path)
    
    from rpyc_classic import ClassicServer
    from rpyc.utils.server import ThreadedServer
    from rpyc import SlaveService

    server = ThreadedServer(SlaveService, port = SERVER_PORT, ipv6 = True)
    print(f"port: {server.port}")
    server.logger.quiet = True
    thread = threading.Thread(target = server.start)
    thread.start()
    
    codec = cv.VideoWriter_fourcc(*"WMV2")
    fps = 12
    out = cv.VideoWriter(output_path, codec , fps, (1920, 1080))
    
    if show_window:
        cv.namedWindow("Recording", cv.WINDOW_NORMAL)
        cv.resizeWindow("Recording", 1920, 1080)
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
    
    out.release() #closing the video file
    if show_window:
        cv.destroyAllWindows() #destroying the recording window
        
    server.close()
    thread.join()
    print("record - server closed")


def play(filepath="T:/Tests/Behave.wmv"):

    cap = cv2.VideoCapture(filepath)

    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    pass
    # record()
