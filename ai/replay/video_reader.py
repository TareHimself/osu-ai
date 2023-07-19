import sys
import json
import threading
import queue
import cv2
import os
import collections
from utils import EventsSampler,Cv2VideoContext

CONFIG_PATH = sys.argv[1]

write_queue = queue.Queue()


# with open(CONFIG_PATH,'r') as f:
#     config_data = json.load(f)

#     sampler = EventsSampler(config_data["events"])

#     iter_target = config_data["iters"]

#     def write_files():
#         global write_queue

#         data = write_queue.get()

#         while data is not None:
#             sample,frame = data
#             text_file_path = os.path.join(config_data['pending_path'],f"{int(sample[0])}.txt")
#             if frame is not None:
#                 cv2.imwrite(os.path.join(config_data['images_path'],f"{int(sample[0])}.png"),frame)
#                 with open(text_file_path,'w') as f:
#                     f.write(f"{sample[0]}|{sample[1]}|{sample[2]}|{sample[3][0]}|{sample[3][1]}")
#             else:
#                 with open(text_file_path,'w') as f:
#                     f.write(f"")

#             data = write_queue.get()
                    


#     write_thread = threading.Thread(target=write_files,group=None,daemon=True)

#     write_thread.start()

#     with Cv2VideoContext(config_data['video_path']) as ctx:
#         for i in iter_target:
#             for i in iter_target:
#                 sample = sampler.sample(i)
#                 cur_time, x, y, keys_bool = sample

#                 ctx.cap.set(
#                     cv2.CAP_PROP_POS_MSEC, cur_time)

#                 read_success, frame = ctx.cap.read()

#                 if read_success:
#                     write_queue.put((sample, frame))

#                 else:

#                     write_queue.put((sample,None))

#     write_queue.put(None)
#     write_queue.join()

with open(CONFIG_PATH,'r') as f:
    config_data = json.load(f)

    sampler = EventsSampler(config_data["events"])

    iter_target: collections.deque = collections.deque(config_data["iters"])

    def write_files():
        global write_queue

        data = write_queue.get()

        while data is not None:
            sample,frame = data
            text_file_path = os.path.join(config_data['pending_path'],f"{int(sample[0])}.txt")
            if frame is not None:
                cur_time, x, y, keys_bool = sample
                image_file_name = f"{config_data['project']}-{round(cur_time)},{1 if keys_bool[0] else 0},{1 if keys_bool[1] else 0},{x},{y}.png"

                image_path = os.path.join(config_data['images_path'],image_file_name)
                
                cv2.imwrite(image_path,frame)

                with open(text_file_path,'w') as f:
                    f.write(f"{image_path}|{image_file_name}")
            else:
                with open(text_file_path,'w') as f:
                    f.write(f"")

            data = write_queue.get()
        
                    


    write_thread = threading.Thread(target=write_files,group=None,daemon=True)

    write_thread.start()

    VIDEO_FRAMES_PER_SECOND = 60
    FPS_MS = VIDEO_FRAMES_PER_SECOND / 1000

    with Cv2VideoContext(config_data['video_path']) as ctx:

        start_idx = iter_target.popleft()

        current_idx = start_idx

        ctx.cap.set(cv2.CAP_PROP_POS_MSEC, current_idx) # set the initial position of the capture

        frames_skipped = 0

        while len(iter_target) > 0 or current_idx is not None:
            sample = sampler.sample(current_idx)
            cur_time, x, y, keys_bool = sample

            ms_diff = current_idx - start_idx

            # ctx.cap.set(cv2.CAP_PROP_POS_MSEC, cur_time)

            if ms_diff != 0:
                
                frames_to_skip = max(int(ms_diff * FPS_MS) - frames_skipped,0)
                skipped = 0
                while skipped != frames_to_skip:
                    ctx.cap.read()
                    skipped += 1
                    frames_skipped += 1
                # print(ms_diff,frames_to_skip,skipped,frames_skipped)


            read_success, frame = ctx.cap.read()
            frames_skipped += 1 # cap.read is the same as frames - 1 so we have to account for that

            if read_success:
                # print(cur_time,ms_diff)
                # cv2.imshow("Window",cv2.circle(frame,(int(x) - 5,int(y) - 5),10,(255,255,255),3))
                # cv2.waitKey(0)
                write_queue.put((sample, frame))
            else:
                write_queue.put((sample,None))

            if len(iter_target) == 0:
                current_idx = None
            else:
                current_idx = iter_target.popleft()
        write_queue.put(None)
        write_thread.join()

