from ai.utils import get_validated_input
from ai.constants import RAW_DATA_DIR
from os import path
from ai.converter import ReplayConverter
import traceback


def start_convert():
    project_name = get_validated_input(
        'What Would You Like To Name This Project ?:', conversion_fn=lambda a: a.lower().strip())
    rendered_path = get_validated_input(
        'Path to the rendered replay video:', validate_fn=lambda a: path.exists(a.strip()),
        conversion_fn=lambda a: a.strip(), on_validation_error=lambda a: print("Invalid path!"))
    replay_json = get_validated_input(
        'Path to the rendered replay json:', validate_fn=lambda a: path.exists(a.strip()),
        conversion_fn=lambda a: a.strip(), on_validation_error=lambda a: print("Invalid path!"))

    num_threads = get_validated_input("Number of threads to use when processing the video (more isn't always faster):",
                                      lambda a: a.strip().isnumeric() and 0 < int(a.strip()), lambda a: int(a.strip()),
                                      on_validation_error=lambda a: print("It must be an integer greater than zero"))

    offset_ms = get_validated_input("Offset in ms to apply to the dataset (e.g. -100):",
                                    lambda a: a.strip().lstrip('-+').isdigit(), lambda a: int(a.strip()),
                                    on_validation_error=lambda a: print("It must be a positive or negative integer"))
    
    max_memory = get_validated_input("Max images to keep in memory when writing. Default is 0 (as much as possible):",
                                    lambda a: (a.strip().lstrip('-+').isdigit() and int(a.strip()) >= 0) if len(a.strip()) > 0 else True, lambda a: int(a.strip()) if len(a.strip()) > 0 else 0,
                                    on_validation_error=lambda a: print("It must be a positive integer or left empty"))

    try:
        ReplayConverter(project_name, rendered_path, replay_json,
                        RAW_DATA_DIR, num_writers=num_threads, frame_offset_ms=offset_ms, max_in_memory=max_memory)
    except:
        traceback.print_exc()
