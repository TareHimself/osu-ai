import collections
from typing import Union
import cv2
import json
from tqdm import tqdm
from queue import Queue
from threading import Thread, Event
import shutil
import os
from ai.utils import Cv2VideoContext, EventsSampler, playfield_coords_to_screen, derive_capture_params
from ai.constants import CAPTURE_HEIGHT_PERCENT


class ReplayConverter:
    """
    NOTE: if the video was shot at 60fps the minimum frame offset would be 
    """

    def __init__(self, project_name: str, danser_video: str, replay_json: str,
                 save_dir: str = "", num_writers=5, max_in_memory=0, frame_interval_ms=10, frame_offset_ms=0,
                 video_fps=100,
                 replay_keys_json: Union[str, None] = None,
                 debug=False) -> None:
        self.project_name = project_name
        self.save_dir = save_dir
        self.danser_video = danser_video
        self.replay_json = replay_json
        self.replay_keys_json = replay_keys_json
        self.num_writers = num_writers
        self.max_in_memory = max_in_memory
        self.frame_interval_ms = frame_interval_ms
        self.frame_offset_ms = frame_offset_ms
        self.video_fps = video_fps
        self.debug = debug
        self.build_dataset()

    def build_dataset(self):

        with Cv2VideoContext(self.danser_video) as ctx:
            screen_height = int(ctx.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            screen_width = int(ctx.cap.get(cv2.CAP_PROP_FRAME_WIDTH))


        [capture_w,capture_h,capture_dx,capture_dy] = derive_capture_params(screen_width,screen_height)

        with open(self.replay_json, 'r') as f:
            replay_data = json.load(f)

        replay_keys_data = None

        if self.replay_keys_json is not None:
            with open(self.replay_keys_json, 'r') as f:
                replay_keys_data = json.load(f)

        start_time = replay_data["objects"][0][
            "start"]  # We assume start time will be the same for both since it should be the same map
        breaks = []

        total_event_time_mouse = 0

        time_offset = start_time + self.frame_offset_ms

        events_keys = []
        events_mouse = []

        for event in replay_data["events"]:
            total_event_time_mouse += event['diff']
            timestamp = total_event_time_mouse - time_offset
            [x,y,dx,dy] = playfield_coords_to_screen(event["x"],event["y"],screen_width,screen_height,True)
            events_mouse.append({
                "x": round(x + dx),
                "y": round(y + dy),
                "time": timestamp,
            })

            if replay_keys_data is None:
                events_keys.append({
                    "keys": [event['k1'], event['k2']],
                    "time": timestamp,
                })

        if replay_keys_data is not None:
            total_event_time_keys = 0
            for event in replay_keys_data["events"]:
                total_event_time_keys += event['diff']
                timestamp = total_event_time_mouse - time_offset
                events_keys.append({
                    "keys": [event['k1'], event['k2']],
                    "time": timestamp,
                })

        for b in replay_data['breaks']:
            breaks.append({
                "start": b["start"] - time_offset,
                "end": b["end"] - time_offset
            })

        stop_time = max(int(events_mouse[len(events_mouse) - 1]['time']),
                        int(events_keys[len(events_keys) - 1]['time']))

        iter_target = range(0, stop_time, self.frame_interval_ms)

        remove_breaks = False

        if remove_breaks:
            new_iter_target = []
            for i in iter_target:
                should_add = True
                for item in breaks:
                    if item['start'] <= i <= item["end"]:
                        should_add = False
                        break

                if should_add:
                    new_iter_target.append(i)
            iter_target = new_iter_target

        save_dir = os.path.join(self.save_dir, self.project_name)

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        os.mkdir(save_dir)

        loading_bar = tqdm(desc="Generating Dataset", total=len(iter_target))

        write_buff = Queue(maxsize=self.max_in_memory)

        stop_conversion = Event()

        def write_one_frame_func(data):
            sample, frame = data
            if frame is not None:
                cur_time, x, y, keys_bool = sample
                image_file_name = f"{self.project_name}-{round(cur_time)},{1 if keys_bool[0] else 0},{1 if keys_bool[1] else 0},{x},{y}.png"

                image_path = os.path.join(save_dir, image_file_name)

                cv2.imwrite(image_path, frame)

                loading_bar.update()

        def frame_writer_func():
            to_write = write_buff.get()
            while to_write is not None and not stop_conversion.is_set():
                write_one_frame_func(to_write)
                to_write = write_buff.get()

        def frame_reader_func(target: list):

            mouse_sampler = EventsSampler(events_mouse.copy())
            keys_sampler = EventsSampler(events_keys.copy())

            local_iter_target: collections.deque = collections.deque(target)

            frame_delta_ms = (1 / self.video_fps) * 1000

            with Cv2VideoContext(self.danser_video) as video_capture_context:

                video_start_time = 0

                target_time = local_iter_target.popleft()

                if self.debug:
                    print("Set time to", target_time)

                total_frames_skipped = 0

                while (len(local_iter_target) > 0 or target_time is not None) and not stop_conversion.is_set():

                    total_time_delta = target_time - video_start_time

                    target_frame = total_time_delta / frame_delta_ms if total_time_delta != 0 else 0

                    frames_to_skip_start = (target_frame - total_frames_skipped)

                    if self.debug:
                        print("INFO", total_time_delta, target_time, target_frame,
                              frame_delta_ms, total_frames_skipped, frames_to_skip_start)

                    if frames_to_skip_start == 0 or frames_to_skip_start >= 1:

                        frames_to_skip = frames_to_skip_start

                        if frames_to_skip > 1:  # no point in skipping frames if we cant read what we have left
                            while frames_to_skip - 1 >= 0:
                                video_capture_context.cap.read()
                                frames_to_skip -= 1
                                if self.debug:
                                    print("SKIPPED")

                        read_success, frame = video_capture_context.cap.read()

                        if read_success:

                            current_time = ((total_frames_skipped + abs(
                                frames_to_skip_start - frames_to_skip)) * frame_delta_ms) + video_start_time
                            
                            cur_time_mouse, x, y = mouse_sampler.sample_mouse(current_time)
                            cur_time_keys, keys_bool = keys_sampler.sample_keys(current_time)

                            if self.debug:
                                print("Key State", keys_bool)

                                # [x,y,dx,dy] = playfield_coords_to_screen(x,y)
                                # debug_x, debug_y = round(
                                #     x + dx), round(
                                #     y + dy),

                                debug_frame = frame[int(capture_dy):int(
                                    capture_dy + capture_h), int(capture_dx):int(
                                    capture_dx + capture_w)].copy()
                                
                                cv2.imshow("Window",
                                           cv2.circle(debug_frame, (int(x) - 5, int(y) - 5), 10,
                                                      (255, 255, 255),
                                                      3))
                                cv2.waitKey(0)

                            write_buff.put(
                                ((current_time, round(x), round(y), keys_bool), frame[int(capture_dy):int(capture_dy + capture_h), int(capture_dx):int(capture_dx + capture_w)]))

                            frames_to_skip -= 1
                        else:
                            loading_bar.update()

                        total_frames_skipped += abs(frames_to_skip_start - frames_to_skip)
                        if self.debug:
                            print("FINAL FRAMES SKIPPED", total_frames_skipped,
                                  frames_to_skip_start, frames_to_skip)
                    else:
                        loading_bar.update()

                    if len(local_iter_target) == 0:
                        target_time = None
                    else:
                        target_time = local_iter_target.popleft()

        frame_reader_thread = Thread(target=frame_reader_func, group=None, daemon=True, args=[iter_target])

        frame_writers = [Thread(target=frame_writer_func, group=None, daemon=True) for x in
                         range(self.num_writers * 2)]  # not sure if 1:2 ratio is right

        frame_reader_thread.start()

        for writer in frame_writers:
            writer.start()

        while frame_reader_thread.is_alive():
            try:
                frame_reader_thread.join(timeout=2)
            except KeyboardInterrupt:
                stop_conversion.set()
                for _ in frame_writers:
                    write_buff.put(None)
                break

        if not stop_conversion.is_set():
            for _ in frame_writers:
                write_buff.put(None)

            for writer in frame_writers:
                while writer.is_alive() and not stop_conversion.is_set():
                    try:
                        writer.join(timeout=2)
                    except KeyboardInterrupt:
                        stop_conversion.set()
                        break
