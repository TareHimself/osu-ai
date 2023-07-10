

import cv2
import sys
import os
import subprocess

class Cv2VideoContext:

    def __init__(self, file_path: str):
        # example file or database connection
        self.file_path = file_path
        self.cap = cv2.VideoCapture(file_path,cv2.CAP_FFMPEG)

    def __enter__(self):
        if self.cap.isOpened() == False:
            raise BaseException(f"Error opening video stream or file {self.file_path}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()


class EventsSampler:
    def __init__(self, events: list) -> None:
        self.events = sorted(events, key=lambda a: a['time'])
        self.events_num = len(self.events)
        self.last_sampled_time = 0
        self.last_sampled_index = 0

    def get(self,idx: int):
        return self.events[idx]['time'],self.events[idx]['x'],self.events[idx]['y'],self.events[idx]['keys']
    
    def sample(self, target_time_ms: float = 0):
        if target_time_ms <= self.events[0]['time']:
            return self.events[0]

        if target_time_ms >= self.events[self.events_num - 1]['time']:
            return self.events[self.events_num - 1]
        
        # search_range = range(self.last_sampled_index,self.events_num - 1) if self.last_sampled_time <= target_time_ms else reversed(range(self.last_sampled_index + 1,0))
        # for i in search_range:
        search_range = range(0,self.events_num - 1)
        for i in search_range:
            event_time, event_x, event_y, event_keys = self.get(i)
            next_event_time, next_event_x, next_event_y, next_event_keys = self.get(i + 1)
            if event_time <= target_time_ms <= next_event_time:
                events_dist = (next_event_time - event_time)

                target_time_dist = (target_time_ms - event_time)
                alpha = target_time_dist / events_dist
                self.last_sampled_index = i
                self.last_sampled_time = target_time_ms
                return event_time + (events_dist * alpha), int(event_x + ((next_event_x - event_x) * alpha)), int(event_y + ((next_event_y - event_y) * alpha)), next_event_keys if next_event_keys[0] or next_event_keys[1] else event_keys
            
        raise BaseException("NO SAMPLE FOUND")
    
def run_file(file_path: str):
    process = subprocess.Popen(f"{sys.executable} {file_path}", shell=True)
    process.communicate()
    return process.returncode