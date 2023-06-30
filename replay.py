import cv2
import zipfile
import json
from os import path
from tqdm import tqdm
from queue import Queue
from threading import Thread


class Cv2VideoContext:

    def __init__(self, file_path: str):
        # example file or database connection
        self.cap = cv2.VideoCapture(file_path)

    def __enter__(self):
        if self.cap.isOpened() == False:
            raise BaseException("Error opening video stream or file")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()


class EventsSampler:
    def __init__(self, events: list) -> None:
        self.events = sorted(events, key=lambda a: a[0])
        self.events_num = len(self.events)

    def sample(self, target_time_ms: float = 0):
        if target_time_ms <= self.events[0][0]:
            return self.events[0]

        if target_time_ms >= self.events[self.events_num - 1][0]:
            return self.events[self.events_num - 1]

        for i in range(self.events_num - 1):
            event_time, event_x, event_y, event_keys = self.events[i]
            next_event_time, next_event_x, next_event_y, next_event_keys = self.events[i + 1]
            if event_time <= target_time_ms <= next_event_time:
                events_dist = (next_event_time - event_time)

                target_time_dist = (target_time_ms - event_time)
                alpha = target_time_dist / events_dist

                return event_time + (events_dist * alpha), int(event_x + ((next_event_x - event_x) * alpha)), int(event_y + ((next_event_y - event_y) * alpha)), next_event_keys if alpha > 0.5 else event_keys


class DatasetCreator:
    def __init__(self, project_name: str, replay_video: str, replay_json: str, save_dir: str = "") -> None:
        self.project_name = project_name
        self.save_dir = save_dir
        self.debug = True
        self.build_dataset(replay_video, replay_json, save_dir)

    def keys(self, n):
        k1 = n & 5 == 5
        k2 = n & 10 == 10
        m1 = not k1 and n & 1 == 1
        m2 = not k2 and n & 2 == 2
        smoke = n & 16 == 16
        return k1 or m1, k2 or m2

    def build_dataset(self, replay_video: str, replay_json: str, save_dir: str):
        with Cv2VideoContext(replay_video) as ctx:
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
            # # Get the total number of frames in the video
            # total_frames = int(ctx.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # # Get the frames per second (FPS) of the video
            # fps = ctx.cap.get(cv2.CAP_PROP_FPS)

            # # Calculate the duration of the video in milliseconds
            # duration_ms = (total_frames / fps) * 1000

            with open(replay_json, 'r') as rep:
                json_data = json.load(rep)

                events = json_data['events']
                first_event_time = events[0]['time']
                converted_events = []

                breaks = []
                for b in json_data['breaks']:
                    breaks.append({
                        'Start': b['Start'] - first_event_time,
                        'End': b['End'] - first_event_time
                    })

                for event in events:
                    x = round(
                        (event['x'] * PLAYFIELD_FACTORY_TO_SCREEN_WIDTH_FACTOR) + PLAYFIELD_OFFSET_X)
                    y = round(
                        (event['y'] * PLAYFIELD_FACTORY_TO_SCREEN_HEIGHT_FACTOR) + PLAYFIELD_OFFSET_Y)

                    keys_bool = self.keys(event['keys'])

                    converted_events.append(
                        (event['time'] - first_event_time, x, y, keys_bool))

                sampler = EventsSampler(converted_events)
                first_event_time = sampler.events[0][0]
                ms_to_sample = int(
                    converted_events[len(converted_events) - 1][0])

                with zipfile.ZipFile(path.join(self.save_dir, f"{self.project_name}.zip"), "w", zipfile.ZIP_DEFLATED) as zip:
                    writer_buff = Queue()

                    def writer():
                        data = writer_buff.get()
                        while data is not None:
                            sample, frame = data
                            cur_time, x, y, keys_bool = sample
                            is_success, buffer = cv2.imencode(
                                ".png", frame)

                            if is_success:
                                zip.writestr(
                                    f"{self.project_name}-{round(cur_time)},{1 if keys_bool[0] else 0},{1 if keys_bool[1] else 0},{x},{y}.png", buffer)

                            data = writer_buff.get()

                    writer_thread = Thread(
                        target=writer, group=None, daemon=True)
                    writer_thread.start()

                    iteration_target = range(0, ms_to_sample, 100)
                    new_iteration_target = []
                    for i in iteration_target:
                        should_add = True
                        for item in breaks:
                            if item['Start'] <= i <= item["End"]:
                                should_add = False
                                break
                        if should_add:
                            new_iteration_target.append(i)

                    for i in tqdm(new_iteration_target, desc="Generating and Compressing Dataset"):

                        sample = sampler.sample(i)
                        cur_time, x, y, keys_bool = sample
                        # cur_time -= first_event_time

                        if cur_time > 0:

                            ctx.cap.set(cv2.CAP_PROP_POS_MSEC, cur_time)

                            read_success, frame = ctx.cap.read()

                            if read_success:

                                writer_buff.put((sample, frame))

                                # cv2.rectangle(frame, (round(PLAYFIELD_OFFSET_X), round(PLAYFIELD_OFFSET_Y)), (round(
                                #     PLAYFIELD_OFFSET_X + PLAYFIELD_SCREEN_WIDTH), round(PLAYFIELD_OFFSET_Y + PLAYFIELD_SCREEN_HEIGHT)), (255, 0, 0), 2)
                                # cv2.circle(frame, (x - 10, y + 10),
                                #            10, (255, 255, 255), 2)

                                # cv2.imshow('EVENT FRAME', frame)
                                # cv2.waitKey(100)
                                # print(x, y, keys_bool, cur_time)
                    print("Waiting for remaining files to save")
                    writer_buff.put(None)
                    writer_thread.join()
                    print("Done")


DatasetCreator("Body Floating (moonshine skin)",
               "replay.mkv", 'replay.json')
