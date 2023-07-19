from math import ceil
import cv2
import zipfile
import json
from tqdm import tqdm
from queue import Queue
from threading import Thread
from tempfile import TemporaryDirectory
import os
from .utils import Cv2VideoContext, run_file


class ReplayConverter:
    def __init__(self, project_name: str, danser_video: str, replay_json: str,save_dir: str = "", num_readers=5, frame_interval_ms=10,frame_offset_ms = 0) -> None:
        self.project_name = project_name
        self.save_dir = save_dir
        self.danser_video = danser_video
        self.replay_json = replay_json
        self.num_readers = num_readers
        self.frame_interval_ms = frame_interval_ms
        self.frame_offset_ms = frame_offset_ms
        self.debug = True
        self.build_dataset()

    def keys(self, n):
        k1 = n & 5 == 5
        k2 = n & 10 == 10
        m1 = not k1 and n & 1 == 1
        m2 = not k2 and n & 2 == 2
        smoke = n & 16 == 16
        return k1 or m1, k2 or m2

    def build_dataset(self):
        with TemporaryDirectory() as working_dir:
            images_path = os.path.join(working_dir,"images")
            pending_path = os.path.join(working_dir,'pending')
            get_config = lambda a: os.path.join(working_dir,f"config-{a}.json")

            os.mkdir(images_path)
            os.mkdir(pending_path)


            SCREEN_HEIGHT = 0
            SCREEN_WIDTH = 0

            with Cv2VideoContext(self.danser_video) as ctx:
                SCREEN_HEIGHT = int(ctx.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                SCREEN_WIDTH = int(ctx.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            PLAYFIELD_RATIO = 4/3
            PLAYFIELD_FACTORY_WIDTH = 512
            PLAYFIELD_FACTORY_HEIGHT = PLAYFIELD_FACTORY_WIDTH / PLAYFIELD_RATIO
            PLAYFIELD_SCREEN_HEIGHT = 0.8 * SCREEN_HEIGHT
            PLAYFIELD_SCREEN_WIDTH = PLAYFIELD_RATIO * PLAYFIELD_SCREEN_HEIGHT
            PLAYFIELD_FACTORY_TO_SCREEN_HEIGHT_FACTOR = PLAYFIELD_SCREEN_HEIGHT / \
                PLAYFIELD_FACTORY_HEIGHT
            PLAYFIELD_FACTORY_TO_SCREEN_WIDTH_FACTOR = PLAYFIELD_SCREEN_WIDTH / \
                PLAYFIELD_FACTORY_WIDTH
            PLAYFIELD_OFFSET_X = (SCREEN_WIDTH - PLAYFIELD_SCREEN_WIDTH) / 2
            PLAYFIELD_OFFSET_Y = (SCREEN_HEIGHT - PLAYFIELD_SCREEN_HEIGHT) / 2

            with open(self.replay_json) as f:
                data = json.load(f)
                start_time = data["start"]
                events = []
                breaks = []

                total_event_time = 0

                time_offset = start_time + self.frame_offset_ms

                for event in data["events"]:
                    total_event_time += event['diff']

                    events.append({
                        "x": round((event["x"] * PLAYFIELD_FACTORY_TO_SCREEN_WIDTH_FACTOR) + PLAYFIELD_OFFSET_X),
                        "y": round((event["y"] * PLAYFIELD_FACTORY_TO_SCREEN_HEIGHT_FACTOR) + PLAYFIELD_OFFSET_Y),
                        "keys": [event['k1'],event['k2']],
                        "time": total_event_time - time_offset
                    })

                for b in data['breaks']:
                    breaks.append({
                        "start": b["start"] - time_offset,
                        "end": b["end"] - time_offset
                    })

                stop_time = int(events[len(events) - 1]['time'])

                iter_target = range(0, stop_time, self.frame_interval_ms)

                remove_breaks = True

                if remove_breaks:
                    new_iter_target = []
                    for i in iter_target:
                        should_add = True
                        for item in breaks:
                            if item['start'] <= i <= item["end"]:
                                should_add = False
                                break
                        # if i == 0:
                        #     print("ZERO ELEMENT", should_add)
                        if should_add:
                            new_iter_target.append(i)
                    iter_target = new_iter_target


                num_readers = self.num_readers
                ammount_per_section = ceil(len(iter_target) / num_readers)
                
                iter_targets =  [iter_target[x * ammount_per_section : (x * ammount_per_section) + ammount_per_section] for x in range(num_readers)]

                

                frame_readers_status = [False for x in range(len(iter_targets))]

                def run_reader(idx: int):
                    nonlocal frame_readers_status

                    file_path = os.path.dirname(os.path.realpath(__file__))

                    run_file(f"{os.path.join(file_path,'video_reader.py')} {get_config(idx)}")

                    frame_readers_status[idx] = True

                frame_readers = [Thread(target=run_reader,group=None,daemon=True,args=[x]) for x in range(len(iter_targets))]
                
                for x in range(len(iter_targets)):
                    with open(get_config(x),'w') as f:
                        json.dump({
                            "project": self.project_name,
                            "events": events,
                            "breaks": breaks,
                            "iters": iter_targets[x],
                            "pending_path": pending_path,
                            "images_path": images_path,
                            "video_path": self.danser_video
                        },f)

                # print("Performing operations in",working_dir)
                
                with zipfile.ZipFile(os.path.join(self.save_dir, f"{self.project_name}.zip"), "w", zipfile.ZIP_DEFLATED) as zip:

                    loadng_bar = tqdm(desc="Generating Dataset", total=len(iter_target))

                    zip_buff = Queue()

                    def write_to_zip():
                            nonlocal zip_buff
                            nonlocal zip
                            nonlocal loadng_bar
                            data = zip_buff.get()
                            while data is not None:
                                file_path, file_name = data

                                zip.write(file_path, file_name)

                                loadng_bar.update()

                                data = zip_buff.get()

                    zip_thread = Thread(target=write_to_zip, group=None, daemon=True)

                    for reader in frame_readers:
                        reader.start()

                    zip_thread.start()

                    while False in frame_readers_status:
                        for file in os.listdir(pending_path):
                            file_path = os.path.join(pending_path,file)

                            try:
                                os.rename(file_path, file_path) ## confirm we have access
                                with open(file_path,'r') as f:
                                    data = f.read().strip()
                                    if len(data) == 0:
                                        loadng_bar.update()
                                    else:
                                        zip_buff.put(data.split("|"))
                                os.remove(file_path)
                            except OSError as e:
                                pass
                    
                    zip_buff.put(None)
                    zip_thread.join()

                   
                    
