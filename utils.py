import asyncio
from os import listdir, path, getcwd
import os
import socket
from threading import Thread, Timer, Event
import time
import traceback
from typing import Callable
import uuid
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


def get_models(prefix="") -> list[str]:
    models = listdir(path.normpath(path.join(
        getcwd(), 'models')))

    filtered = []

    for model in models:
        if model.startswith(prefix):
            filtered.append(model)

    return filtered


def get_datasets() -> list[str]:
    return listdir(RAW_DATA_DIR)


def get_model_path(model):
    return path.normpath(path.join(
        getcwd(), 'models', model))


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
        self.active = True

    def connect(self):
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
        while True:
            try:
                message, address = self.socket.recvfrom(1024)
                message = message.decode("utf-8")
                self.on_message_internal(message)
            except socket.timeout:
                break

        self.socket.close()

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
        if not self.active:
            self.socket.settimeout(1)
            self.socket.close()
        self.active = True


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
