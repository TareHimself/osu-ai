import asyncio
import json
import os
import socket
import time
import traceback
import cv2
import numpy as np
from os import listdir, path
from socket import SHUT_RDWR
from tempfile import TemporaryDirectory
from threading import Thread, Timer, Event
from datetime import datetime
from ai.constants import RAW_DATA_DIR, MODELS_DIR,CAPTURE_HEIGHT_PERCENT
from mss import mss
from queue import Queue
from tqdm import tqdm
from ai.enums import EModelType
from typing import TypeVar, Callable, Union, TypeVar
import math
import sys
import subprocess


class Cv2VideoContext:

    def __init__(self, file_path: str):
        # example file or database connection
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path, cv2.CAP_FFMPEG)

    def __enter__(self):
        if not self.cap.isOpened():
            raise BaseException(f"Error opening video stream or file {self.file_path}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()


T = TypeVar("T")
Z = TypeVar("Z")


class EventsSamplerEventTypes:
    MOUSE = 0
    KEYS = 1


class EventsSampler:
    def __init__(self, events: list[T]) -> None:
        self.events = sorted(events, key=lambda a: a['time'])
        self.events_num = len(self.events)
        self.last_sampled_time = 0
        self.last_sampled_index = 0

    def get(self, idx: int):
        return self.events[idx]['time'], self.events[idx]['x'], self.events[idx]['y'], self.events[idx]['keys']

    def sample_mouse(self, target_time_ms: float = 0) -> Z:
        if target_time_ms <= self.events[0]['time']:
            return self.events[0]

        if target_time_ms >= self.events[self.events_num - 1]['time']:
            return self.events[self.events_num - 1]

        # search_range = range(self.last_sampled_index,self.events_num - 1) if self.last_sampled_time <= target_time_ms else reversed(range(self.last_sampled_index + 1,0))
        # for i in search_range:
        search_range = range(0, self.events_num - 1)
        for i in search_range:
            cur_time = self.events[i]['time']
            next_time = self.events[i + 1]['time']
            if cur_time <= target_time_ms <= next_time:
                events_dist = (next_time - cur_time)

                target_time_dist = (target_time_ms - cur_time)
                alpha = target_time_dist / events_dist
                a = self.events[i]
                b = self.events[i + 1]
                self.last_sampled_index = i
                self.last_sampled_time = target_time_ms
                return cur_time, a["x"] + ((b['x'] - a['x']) * alpha), a["y"] + ((b['y'] - a['y']) * alpha)

        raise BaseException("NO SAMPLE FOUND")

    def sample_keys(self, target_time_ms: float = 0) -> Z:
        if target_time_ms <= self.events[0]['time']:
            return self.events[0]

        if target_time_ms >= self.events[self.events_num - 1]['time']:
            return self.events[self.events_num - 1]

        # search_range = range(self.last_sampled_index,self.events_num - 1) if self.last_sampled_time <= target_time_ms else reversed(range(self.last_sampled_index + 1,0))
        # for i in search_range:
        search_range = range(0, self.events_num - 1)
        for i in search_range:
            cur_time = self.events[i]['time']
            next_time = self.events[i + 1]['time']
            if cur_time <= target_time_ms <= next_time:
                events_dist = (next_time - cur_time)

                target_time_dist = (target_time_ms - cur_time)
                alpha = target_time_dist / events_dist
                a = self.events[i]
                b = self.events[i + 1]
                self.last_sampled_index = i
                self.last_sampled_time = target_time_ms
                return cur_time, b["keys"] if alpha >= 0.5 else a['keys']

        raise BaseException("NO SAMPLE FOUND")


class KeysSampler:
    def __init__(self, keys_events: list) -> None:
        self.events = sorted(keys_events, key=lambda a: a['time'])
        self.events_num = len(self.events)
        self.last_sampled_time = 0
        self.last_sampled_index = 0

    def get(self, idx: int):
        return self.events[idx]['time'], self.events[idx]['keys']

    def sample(self, target_time_ms: float = 0, key_press_allowance_ms=6) -> list[float, tuple]:
        if target_time_ms <= self.events[0]['time']:
            return self.events[0]

        if target_time_ms >= self.events[self.events_num - 1]['time']:
            return self.events[self.events_num - 1]

        # search_range = range(self.last_sampled_index,self.events_num - 1) if self.last_sampled_time <= target_time_ms else reversed(range(self.last_sampled_index + 1,0))
        # for i in search_range:
        search_range = range(0, self.events_num - 1)
        last_idx_with_press = -1
        for i in search_range:
            event_time, event_keys = self.get(i)
            next_event_time, next_event_keys = self.get(i + 1)
            if event_keys[0] or event_keys[1]:
                last_idx_with_press = i
            if event_time <= target_time_ms <= next_event_time:
                events_dist = (next_event_time - event_time)

                target_time_dist = (target_time_ms - event_time)
                alpha = target_time_dist / events_dist
                self.last_sampled_index = i
                self.last_sampled_time = target_time_ms

                keys_result = (False, False)

                next_idx_with_press = -1
                for j in range(i, self.events_num - 1):
                    cur = self.get(j)
                    if cur[1][0] or cur[1][1]:
                        next_idx_with_press = j
                        break

                dist_last_press = target_time_ms - self.get(last_idx_with_press)[0]
                if next_idx_with_press != -1:

                    dist_next_press = self.get(next_idx_with_press)[0] - target_time_ms

                    if dist_last_press < dist_next_press:
                        if dist_last_press <= key_press_allowance_ms:
                            keys_result = self.get(last_idx_with_press)[1]
                    else:
                        if dist_next_press <= key_press_allowance_ms:
                            keys_result = self.get(next_idx_with_press)[1]

                elif dist_last_press <= key_press_allowance_ms:
                    keys_result = self.get(last_idx_with_press)[1]

                return [event_time + (events_dist * alpha), keys_result]

        raise BaseException("NO SAMPLE FOUND")


def run_file(file_path: str):
    process = subprocess.Popen(f"{sys.executable} {file_path}", shell=True)
    process.communicate()
    return process.returncode

def derive_capture_params(window_width=1920, window_height=1080):
    osu_play_field_ratio = 3 / 4
    capture_height= int(window_height * CAPTURE_HEIGHT_PERCENT)
    capture_width = int(capture_height / osu_play_field_ratio)
    capture_params = [capture_width, capture_height,
                      int((window_width - capture_width) / 2), int((window_height - capture_height) / 2)]

    return capture_params

def playfield_coords_to_screen(playfield_x,playfield_y,screen_w=1920,screen_h=1080,account_for_capture_params = False):
    

    play_field_ratio = 4 / 3
    screen_ratio = screen_w  / screen_h

    play_field_factory_width = 512
    play_field_factory_height = play_field_factory_width / play_field_ratio
    factory_h = play_field_factory_height * 1.2
    factory_w = factory_h * screen_ratio
    factory_dx = (factory_w - play_field_factory_width) / 2
    factory_dy = (factory_h - play_field_factory_height) / 2
    screen_dx = factory_dx * (screen_w / factory_w)
    screen_dy = factory_dy * (screen_h / factory_h)
    screen_x = playfield_x * (screen_w / factory_w)
    screen_y = playfield_y * (screen_h / factory_h)

    if account_for_capture_params:
        cap_x,cap_y,cap_dx,cap_dy = derive_capture_params(screen_w,screen_h)
        screen_dx = screen_dx - cap_dx
        screen_dy = screen_dy - cap_dy

    return [screen_x,screen_y,screen_dx,screen_dy]


"""
    Ensures this context runs for the given fixed time or more

    Returns:
        _type_: _description_

    """


class FixedRuntime:
    def __init__(self, target_time: float = 1.0, debug=None):
        self.start = 0
        self.delay = target_time
        self.debug = debug

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        elapsed = time.time() - self.start
        wait_time = self.delay - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
            if self.debug is not None:
                print(f"Context [{self.debug}] elapsed {wait_time * -1:.4f}s")
        else:
            if self.debug is not None:
                print(f"Context [{self.debug}] elapsed {wait_time * -1:.4f}s")


MESSAGES_SENT = 0

AIM_MODELS = []
CLICKS_MODELS = []
COMBINED_MODELS = []


def refresh_model_list():
    global AIM_MODELS
    global CLICKS_MODELS
    global COMBINED_MODELS
    AIM_MODELS = []
    CLICKS_MODELS = []
    COMBINED_MODELS = []
    for model_id in os.listdir(MODELS_DIR):

        model_path = os.path.join(MODELS_DIR, model_id)

        with open(os.path.join(model_path, 'info.json'), 'r') as f:
            data = json.load(f)

            payload = {
                'id': model_id,
                'name': data['name'],
                'date': datetime.strptime(data['date'], "%Y-%m-%d %H:%M:%S.%f"),
                'channels': data['channels'],
                'datasets': data['datasets']
            }

            if data['type'] == EModelType.Aim.value:
                AIM_MODELS.append(payload)
            elif data['type'] == EModelType.Actions.value:
                CLICKS_MODELS.append(payload)
            elif data['type'] == EModelType.Combined.value:
                COMBINED_MODELS.append(payload)

    AIM_MODELS = sorted(AIM_MODELS, key=lambda a: 0 - a['date'].timestamp())
    CLICKS_MODELS = sorted(CLICKS_MODELS, key=lambda a: 0 - a['date'].timestamp())
    COMBINED_MODELS = sorted(COMBINED_MODELS,
                             key=lambda a: 0 - a['date'].timestamp())


refresh_model_list()


def get_models(model_type: EModelType) -> list[dict]:
    global AIM_MODELS
    global CLICKS_MODELS
    global COMBINED_MODELS
    if model_type == EModelType.Aim:
        return AIM_MODELS
    elif model_type == EModelType.Actions:
        return CLICKS_MODELS
    else:
        return COMBINED_MODELS


def get_datasets() -> list[str]:
    return listdir(RAW_DATA_DIR)


def get_validated_input(prompt="You forgot to put your own prompt",
                        validate_fn: Callable[[str], bool] = lambda a: len(a.strip()) != 0,
                        conversion_fn: Callable[[str], T] = lambda a: a.strip(),
                        on_validation_error: Callable[[str], None] = lambda
                                a: print("Invalid input, please try again.")) -> T:
    input_as_str = input(prompt)

    if not validate_fn(input_as_str):
        on_validation_error(input_as_str)
        return get_validated_input(prompt, validate_fn, conversion_fn)

    return conversion_fn(input_as_str)


class FileWatcher(Thread):
    def __init__(self, file_path, callback, poll_frequency=0.05):
        super().__init__(group=None, daemon=True)
        self.file_path = file_path
        self.callback = callback
        self.freq = poll_frequency
        self.callback(open(self.file_path).readlines())
        self.buff = Queue()
        self.start()

    def kill(self):
        if self.is_alive():
            self.buff.put("Shinu")
            self.join()

    def run(self):
        modified_on = os.path.getmtime(self.file_path)
        try:
            while True:
                if not self.buff.empty():
                    break
                time.sleep(self.freq)
                modified = os.path.getmtime(self.file_path)
                if modified != modified_on:
                    modified_on = modified
                    self.callback(open(self.file_path).readlines())
        except Exception as e:
            print(traceback.format_exc())


class OsuSocketServer:
    def __init__(self, on_state_updated) -> None:
        self.active_thread = None
        self.osu_game = None
        self.active = False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.on_state_updated = on_state_updated
        self.pending_messages = {}

    def connect(self):
        self.active = True
        self.sock.bind(("127.0.0.1", 11000))
        self.osu_game = ("127.0.0.1", 12000)
        self.active_thread = Thread(group=None, target=self.receive_messages, daemon=True)
        self.active_thread.start()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.kill()

    def on_message_internal(self, message):
        message_id, content = message.split('|')
        if content == "MAP_BEGIN" or content == "MAP_END":
            self.on_state_updated(content)
            return
        # print("<<", content)
        if message_id in self.pending_messages.keys():
            task, loop, timr = self.pending_messages[message_id]
            loop.call_soon_threadsafe(task.set_result, content)
            del self.pending_messages[message_id]
            timr.cancel()

    def receive_messages(self):
        while self.active:
            try:
                if self.sock is not None:
                    message, address = self.sock.recvfrom(1024)
                    message = message.decode("utf-8")
                    self.on_message_internal(message)
            except socket.timeout:
                break

    def send(self, message: str):
        self.sock.sendto(
            f"NONE|{message}".encode("utf-8"), self.osu_game)

    def cancel_send_and_wait(self, m_id, value):
        if m_id in self.pending_messages.keys():
            task, loop, timr = self.pending_messages[m_id]
            loop.call_soon_threadsafe(task.set_result, value)
            del self.pending_messages[m_id]

    async def send_and_wait(self, message: str, timeout_value="", timeout=10):
        global MESSAGES_SENT
        loop = asyncio.get_event_loop()
        task = asyncio.Future()
        message_id = f"{MESSAGES_SENT}"
        MESSAGES_SENT += 1
        self.pending_messages[message_id] = task, loop, Timer(timeout, self.cancel_send_and_wait, [
            message_id, timeout_value])
        self.pending_messages[message_id][2].start()
        self.sock.sendto(
            f"{message_id}|{message}".encode("utf-8"), self.osu_game)
        result = await task
        return result

    def kill(self):
        if self.active:
            target = self.sock
            self.sock = None
            target.settimeout(1)
            target.shutdown(SHUT_RDWR)
            target.close()
        self.active = False


class ScreenRecorder(Thread):
    def __init__(self, fps: int = 30):
        super().__init__(group=None, daemon=True)
        self.fps = fps
        self.stop_event = Event()
        self.start()

    def stop(self):
        self.stop_event.set()

    def run(self):
        filename = f"{int(time.time() * 1000)}.avi"
        write_buff = Queue()
        with TemporaryDirectory() as record_dir:

            def write_frames():
                frames_saved = 0

                frame = write_buff.get()

                while frame is not None:
                    frame = np.array(frame)
                    cv2.imwrite(
                        path.join(record_dir, f'{frames_saved}.png'), frame)
                    frames_saved += 1
                    frame = write_buff.get()

            write_thread = Thread(target=write_frames, group=None, daemon=True)
            write_thread.start()

            with mss() as sct:
                while True:
                    with FixedRuntime(1 / self.fps):
                        write_buff.put(sct.grab(sct.monitors[1]))
                        if self.stop_event.is_set():
                            break

            write_buff.put(None)
            write_thread.join()

            files = os.listdir(record_dir)
            files.sort(key=lambda a: int(a.split('.')[0]))

            source = cv2.VideoWriter_fourcc(*"MJPG")

            writer = cv2.VideoWriter(
                filename, source, float(self.fps), (1920, 1080))

            for file in tqdm(files, desc=f"Generating Video from {len(files)} frames."):
                writer.write(cv2.imread(path.join(record_dir, file)))

            writer.release()
