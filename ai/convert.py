from utils import get_validated_input
from constants import RAW_DATA_DIR
from os import path
from ai.replay.converter import ReplayConverter
import traceback
# list_window_names()


def start_convert():
    project_name = get_validated_input(
        'What Would You Like To Name This Project ?:', conversion_fn=lambda a: a.lower().strip())
    rendered_path = get_validated_input(
        'Path to the rendered replay video:', validate_fn=lambda a: path.exists(a.strip()), conversion_fn=lambda a: a.strip(), validation_error_message_fn=lambda a: "Invalid path!")
    replay_json = get_validated_input(
        'Path to the rendered replay json:', validate_fn=lambda a: path.exists(a.strip()), conversion_fn=lambda a: a.strip(), validation_error_message_fn=lambda a: "Invalid path!")

    num_threads = get_validated_input("Number of processes to use when processing the video (more isn't always faster):",
                                      lambda a: a.strip().isnumeric() and 0 < int(a.strip()), lambda a: int(a.strip()), validation_error_message_fn=lambda a: "It must be an integer greater than zero")
    
    offset_ms = get_validated_input("Offset in ms to apply to the dataset (e.g. -100):",
                                      lambda a: a.strip().lstrip('-+').isdigit(), lambda a: int(a.strip()), validation_error_message_fn=lambda a: "It must be a positive or negative integer")
    
    try:
        ReplayConverter(project_name, rendered_path, replay_json,
                        RAW_DATA_DIR, num_readers=num_threads,frame_offset_ms=offset_ms)
    except Exception as e:

        print(traceback.format_exc())
