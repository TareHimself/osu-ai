import asyncio
import json
from os import listdir, path, getcwd
import os
import socket
from socket import SHUT_RDWR
from threading import Thread, Timer, Event
import time
import traceback
from typing import Callable, Union
from constants import RAW_DATA_DIR
import torch
import cv2
from queue import Queue

from windows import WindowCapture

"""
    Ensures this context runs for the given fixed time or more

    Returns:
        _type_: _description_

    """


class FixedRuntime():
    def __init__(self, target_time=1, debug=None):
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
MODELS_PATH = './models'


def refresh_model_list():
    global AIM_MODELS
    global CLICKS_MODELS
    AIM_MODELS = []
    CLICKS_MODELS = []
    for model_id in os.listdir(MODELS_PATH):

        model_path = os.path.join(MODELS_PATH, model_id)
        with open(os.path.join(model_path, 'info.json'), 'r') as f:
            data = json.load(f)
            payload = (model_id, data['dataset'],
                       data['date'], data['channels'])
            if data['type'] == 'aim':
                AIM_MODELS.append(payload)
            else:
                CLICKS_MODELS.append(payload)


refresh_model_list()


def get_models(model_type: str) -> list[str]:
    global AIM_MODELS
    global CLICKS_MODELS
    if model_type == 'aim':
        return AIM_MODELS
    return CLICKS_MODELS


def get_datasets() -> list[str]:
    return listdir(RAW_DATA_DIR)


def get_validated_input(prompt="You forgot to put your own prompt", validate_fn=lambda a: len(a.strip()) != 0,
                        conversion_fn=lambda a: a.strip()):
    input_as_str = input(prompt)
    if not validate_fn(input_as_str):
        print("Invalid input, please try again.")
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
        modifiedOn = os.path.getmtime(self.file_path)
        try:
            while True:
                if not self.buff.empty():
                    break
                time.sleep(self.freq)
                modified = os.path.getmtime(self.file_path)
                if modified != modifiedOn:
                    modifiedOn = modified
                    self.callback(open(self.file_path).readlines())
        except Exception as e:
            print(traceback.format_exc())


class OsuSocketServer:
    def __init__(self, on_state_updated) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.on_state_updated = on_state_updated
        self.pending_messages = {}

    def connect(self):
        self.active = True
        self.socket.bind(("127.0.0.1", 9200))
        self.client = ("127.0.0.1", 9500)
        self.t1 = Thread(group=None, target=self.recieve_messages, daemon=True)
        self.t1.start()

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
        #print("<<", content)
        if message_id in self.pending_messages.keys():
            task, loop, timr = self.pending_messages[message_id]
            loop.call_soon_threadsafe(task.set_result, content)
            del self.pending_messages[message_id]
            timr.cancel()

    def recieve_messages(self):
        while self.active:
            try:
                if self.socket is not None:
                    message, address = self.socket.recvfrom(1024)
                    message = message.decode("utf-8")
                    self.on_message_internal(message)
            except socket.timeout:
                break

    def send(self, message: str):
        self.socket.sendto(
            f"NONE|{message}".encode("utf-8"), self.client)

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
        self.socket.sendto(
            f"{message_id}|{message}".encode("utf-8"), self.client)
        result = await task
        return result

    def kill(self):
        if self.active:
            target = self.socket
            self.socket = None
            target.settimeout(1)
            target.shutdown(SHUT_RDWR)
            target.close()
        self.active = False


class RecorderThread(Thread):
    def __init__(self, rate: int):
        super().__init__(group=None, daemon=True)
        self.rate = rate
        self.stop_event = Event()
        self.start()

    def stop(self):
        self.stop_event.set()

    def run(self):
        print(self.rate, 1/self.rate)
        source = cv2.VideoWriter_fourcc(*"XVID")
        cap = WindowCapture()
        writer = cv2.VideoWriter(
            f"{int(time.time() * 1000)}.avi", source, self.rate, (1920, 1080))

        while True:
            with FixedRuntime(1 / self.rate, 'Screen'):
                writer.write(cap.capture())
                if self.stop_event.is_set():
                    break

        writer.release()


class ScreenRecorder():
    def __init__(self, rate: int = 30):
        self.rate = rate
        self.thread = None

    def start(self):
        if self.thread is not None:
            self.thread.stop()

        self.thread = RecorderThread(rate=self.rate)

    def stop(self):
        if self.thread is not None:
            self.thread.stop()
            self.thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()
