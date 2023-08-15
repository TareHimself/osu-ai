import os
import zipfile
from tqdm import tqdm

# try:
#     with ScreenRecorder(20) as s:
#         while True:
#             time.sleep(5)

# except Exception as e:
#     print(e)
#     print(traceback.print_exc())


# def load_video(location):
#     vidcap = cv2.VideoCapture(location)
#     return (vidcap, int(os.path.getctime(location) * 1000))


# video, created = load_video('1678631402353.avi')


# def get_frame(time_seconds):
#     global video
#     video.set(cv2.CAP_PROP_POS_MSEC, time_seconds)
#     hasFrames, image = video.read()
#     if hasFrames:
#         return image
#     return None


# record_time = 1678630643467
# cur_time = 1678630643467
# try:
#     while True:
#         cv2.imshow("Debug", get_frame((cur_time - record_time)))
#         cv2.waitKey(100)
#         cur_time += 1

# except Exception as e:
#     print(e)
#     print(traceback.print_exc())


# with open('sample.txt', 'r') as f:
#     for line in f.readlines():
#         timestamp, data = line.split('|')

#         k1, k2, mx, my = data.split(',')

#         mx = int((int(mx) / 1920) * 1280)

#         my = int((int(my) / 1080) * 720)

#         timestamp = int(timestamp)

#         target_secs = (1678631412546 - 1678631402353)
#         print(target_secs)
#         frame = get_frame(target_secs)

#         frame_with_circle = cv2.circle(
#             frame, (mx, my), 20, (255, 0, 0), 10)
#         cv2.imshow("Debug", frame_with_circle)
#         cv2.waitKey(0)


# # sample = get_frame(1*1000)
# # print(created)
# # cv2.imshow("Image", sample)
# # cv2.waitKey(0)

FILES_PATH = os.path.join(os.getcwd(), 'pending-capture')
COMPRESSED_PATH = os.path.join(os.getcwd(), 'source.zip')
zip = zipfile.ZipFile(COMPRESSED_PATH, "w", zipfile.ZIP_DEFLATED)

for file in tqdm(os.listdir(FILES_PATH), desc="Zipping up source files"):
    zip.write(os.path.join(FILES_PATH, file), file)
    # data, ext = file.split('.')
    # projId, data = data.split('-')

    # timestamp, k1, k2, mx, my = data.split(',')

    # mx = int(mx)

    # my = int(my)

    # timestamp = int(timestamp)

    # frame = cv2.imread(os.path.join(FILES_PATH, file))

    # frame_with_circle = cv2.circle(
    #     frame, (mx, my), 20, (255, 0, 0), 10)
    # cv2.imshow("Debug", frame_with_circle)
    # cv2.waitKey(1)

zip.close()
