import os
import time
import traceback
from threading import Thread


class WatchFileContent(Thread):
    def __init__(self, file_path, callback,poll_frequency=0.05):
        super().__init__(group=None, daemon=True)
        self.file_path = file_path
        self.callback = callback
        self.freq = poll_frequency
        self.callback(open(self.file_path).readlines())

    def run(self):
        modifiedOn = os.path.getmtime(self.file_path)
        try:
            while True:
                time.sleep(self.freq)
                modified = os.path.getmtime(self.file_path)
                if modified != modifiedOn:
                    modifiedOn = modified
                    self.callback(open(self.file_path).readlines())
        except Exception as e:
            print(traceback.format_exc())


# def on_file_modified(lines):
#     print(lines)

def on_left_state_modified(lines):
    print("Left Button:","DOWN" if lines[0] == "1" else "UP")
    
def on_right_state_modified(lines):
    print("Right Button:", "DOWN" if lines[0] == "1" else "UP")

# WatchFileContent(r'C:\Users\Taree\Pictures\accuracy.txt',
#                  on_file_modified).start()

WatchFileContent(r'C:\Users\Taree\Pictures\Action RightButton.txt',
                 on_left_state_modified,0.01).start()

WatchFileContent(r'C:\Users\Taree\Pictures\Action LeftButton.txt',
                 on_right_state_modified, 0.01).start()

while True:
    time.sleep(10)
