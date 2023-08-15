import math
import json
from ai.replay.utils import make_keys_json
import osrparse
from osrparse import Replay, parse_replay_data
from ai.replay.beatmap import BeatmapParser

# def add(replay, x, y, key, rtime):
#     r = [None, None, None, None]
#     r[Replays.CURSOR_X] = x
#     r[Replays.CURSOR_Y] = y
#     r[Replays.KEYS_PRESSED] = key
#     r[Replays.TIMES] = rtime
#     replay.append(r)
#
#


with open("PassCode - Ray (Akitoshi) [Extreme].keys.json",'w') as f:
    json.dump(make_keys_json("PassCode - Ray (Akitoshi) [Extreme].osu"), f,indent=3)
