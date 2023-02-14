import asyncio
from os import listdir, path, getcwd
import os
import socket
from threading import Thread
import time
import traceback
from typing import Callable
import uuid
from constants import RAW_DATA_DIR
import torch
from queue import Queue


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


def load_model_data(model):
    return torch.load(path.normpath(path.join(
        getcwd(), 'models', model)))


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
    def __init__(self, on_message=lambda a: a) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", 12000))
        self.client = None
        self.on_message = on_message
        self.pending_messages = {}
        self.active = True
        self.t1 = Thread(group=None, target=self.recieve_messages, daemon=True)
        self.t1.start()

    def on_message_internal(self, message):
        message_id, content = message.split('|')

        if message_id in self.pending_messages.keys():
            task, loop = self.pending_messages[message_id]
            loop.call_soon_threadsafe(task.set_result, content)

    def recieve_messages(self):
        while True:
            try:
                message, address = self.socket.recvfrom(1024)
                self.client = address
                message = message.decode("utf-8")
                self.on_message(message)
                self.on_message_internal(message)
            except socket.timeout:
                break

        self.socket.close()

    def send(self, message: str):

        if self.client is not None:
            self.socket.sendto(
                f"NONE|{message}".encode("utf-8"), self.client)

    async def send_and_wait(self, message: str):
        if self.client is None:
            return None
        loop = asyncio.get_event_loop()
        task = asyncio.Future()
        message_id = str(uuid.uuid4())
        self.pending_messages[message_id] = task, loop
        self.socket.sendto(
            f"{message_id}|{message}".encode("utf-8"), self.client)
        result = await task
        return result

    def kill(self):
        self.socket.settimeout(1)
