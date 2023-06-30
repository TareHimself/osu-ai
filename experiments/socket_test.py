
import socket
import time
from threading import Thread
import uuid
import asyncio

from utils import OsuSocketServer

server = OsuSocketServer()
while True:
    time.sleep(0.01)
    print("\tGame Time: ", asyncio.run(
        server.send_and_wait("time")), "                           ", end="\r") if server.osu_game is not None else print("\tClient hasnt connected                          ", end="\r")
