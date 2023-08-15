import time
import asyncio

from ai.utils import OsuSocketServer

server = OsuSocketServer()
while True:
    time.sleep(0.01)
    print("\tGame Time: ", asyncio.run(
        server.send_and_wait("time")), "                           ", end="\r") if server.osu_game is not None else print("\tClient hasnt connected                          ", end="\r")
