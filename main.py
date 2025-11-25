import os

import sys

import cv2

from datetime import datetime

import time

from perception_engine import PerceptionEngine



# Get the absolute path to the project root (assuming this script is inside the project folder)

project_root = os.path.abspath(os.path.dirname(__file__))



# Add project_root to sys.path if it's not already included

if project_root not in sys.path:

    sys.path.append(project_root)

else:

    print("Path already exists")
    

livefeed_dir = os.path.join(project_root, "livefeed")

os.makedirs(livefeed_dir, exist_ok=True)


import json

from helpers import is_height_greater_than_width, is_width_greater_than_height

INFERENCE_TIMES_FILE = os.path.join(livefeed_dir, "last_inference_time.txt")
inference_times = []




# =============== Need to change it later for the background processing =====================

from threading import Thread

from queue import Queue



# Thread-safe queue to store video save tasks

event_video_queue = Queue()



def write_event_video_worker():

    while True:

        item = event_video_queue.get()

        if item is None:

            break  # Exit signal received



        frames, output_path, fps, frame_size = item

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

        for f in frames:

            out.write(f)

        out.release()

        print(f"[INFO] Event video saved to: {output_path}")

        event_video_queue.task_done()



# Start the writer thread

writer_thread = Thread(target=write_event_video_worker, daemon=True)

writer_thread.start()

# =============== Need to change it later for the background processing =====================







if __name__ == "__main__":



    # Get the directory where the current script is located

    base_dir = os.path.dirname(os.path.abspath(__file__))



    # Construct the path to the config.json

    config_path = os.path.join(base_dir, "config.json")



    # Load the config

    with open(config_path) as f:

        config = json.load(f)

    

    if config.get('video',{}).get('type') == 'rtsp':

        cap = cv2.VideoCapture(config.get("video_path"), cv2.CAP_FFMPEG)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)



    elif config.get('video',{}).get('type') == 'local':

        cap = cv2.VideoCapture(config.get("video",{}).get('url'))



    if not cap.isOpened():

        print("Error: Could not open video stream.")

        exit()



    print("Connected to RTSP stream. Press 'q' to quit.")

    # ---------------- Video Level Variables ----------------



    state = 'start'

    state_tolerance_sec = 2

    start_ts, watchdog_ts, generateEvent_ts, last_event_generated_ts = None, None, None, None

    prev_frame_status, curr_frame_status = None, None 

    is_width_and_height_line_cut_each_other = False

    meta_data_list, events, event_buffer_raw, event_buffer_annotated = [], [], [], []

    frame_width, frame_height = 1280, 720

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')



    # livefeed_dir = os.path.join(project_root, "livefeed")

    # os.makedirs(livefeed_dir, exist_ok=True)



    start_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



    live_feed_full_fps = os.path.join(livefeed_dir, f"{start_time_str}_live_full_fps.mp4")

    live_feed_controlled_fps = os.path.join(livefeed_dir, f"{start_time_str}_live_feed_controlled_fps.mp4")

    annotated_output_path_controlled_fps = os.path.join(livefeed_dir, f"{start_time_str}_annotated_controlled_fps.mp4")



    window_width = config.get("video",{}).get('display',{}).get("width")

    window_height  = config.get("video",{}).get('display',{}).get("height")





    live_feed_full_fps_writer = cv2.VideoWriter(live_feed_full_fps, fourcc, config.get('video',{}).get('fps'), (window_width, window_height)) 

    live_feed_controlled_fps_writer = cv2.VideoWriter(live_feed_controlled_fps, fourcc, config.get('analysis_engine',{}).get('fps'), (window_width, window_height))

    annotated_controlled_fps_writer = cv2.VideoWriter(annotated_output_path_controlled_fps, fourcc, config.get('analysis_engine',{}).get('fps'), (window_width, window_height))



    # fps controller

    time_interval = round( (1 / config.get('video',{}).get('fps') ) , 5)

    prev_ts = time.time()



    # perception engine 

    pe = PerceptionEngine(

        model_name = config.get("perception_engine", {}).get("model_name"),

        model_threshold = config.get("perception_engine", {}).get("confidence_threshold")

        )





    while cap.isOpened():

        ret, frame = cap.read()



        if not ret:

            print("Failed to grab frame.")

            break



        frame_height, frame_width = frame.shape[:2]

        raw_frame = frame.copy()



        print(f"------------------- {time.time()} -------------------")



        if config.get("video",{}).get('display',{}).get("flag"):

            cv2.imshow(

                'RTSP Live Camera Feed', 

                cv2.resize(

                    frame, (window_width, window_height))

                )

            if cv2.waitKey(1) & 0xFF == ord('q'):

                break        

        

        curr_ts = time.time()



        if (curr_ts - prev_ts) >= time_interval:

            prev_ts = curr_ts



            # inference 

            bounding_boxes, confs, inference_time = pe.get_bounding_boxes(image=frame, conf=0.75)
            inference_times.append(inference_time)


            print('[YOLO] Inference Time')



            if len(bounding_boxes) == 0:

                print("[WARNING] No bounding box detected")

                continue



            frame, metadata = pe.draw_bounding_box(frame, boxes=bounding_boxes, confs=confs)



            prev_frame_status = curr_frame_status

            curr_frame_status = is_width_greater_than_height(metadata)



            if prev_frame_status is False and curr_frame_status is True:

                is_width_and_height_line_cut_each_other = True

                start_ts = time.time()



            print(f"prev: {prev_frame_status}, curr: {curr_frame_status}, intersected: {is_width_and_height_line_cut_each_other}")

            meta_data_list.append(metadata)



            # === State Machine ===

            if (last_event_generated_ts is None) or (time.time() - last_event_generated_ts >= config.get('analysis_engine',{}).get('cooldown')): 

                if state == 'start':

                    if is_width_and_height_line_cut_each_other:

                        state = 'watchdog'

                        watchdog_ts = time.time()

                        print("[STATE TRANSITION] START → WATCHDOG")

                        is_width_and_height_line_cut_each_other = False

                    else:

                        if len(event_buffer_raw) > (state_tolerance_sec * config.get('analysis_engine',{}).get('fps')):

                            event_buffer_raw.pop(0)



                            event_buffer_raw.append(raw_frame)

                            

                        else:

                            event_buffer_raw.append(raw_frame)

                            



                elif state == 'watchdog':

                    cv2.rectangle(frame, (0, 0), (frame_width - 1, frame_height - 1), (0, 0, 255), 6)

                    if is_height_greater_than_width(metadata):

                        if (curr_ts - watchdog_ts) >= state_tolerance_sec:

                            state = 'generateEventCheck'

                            generateEvent_ts = curr_ts

                            print("[STATE TRANSITION] WATCHDOG → GENERATE EVENT")

                    

                    else:

                        if (curr_ts - watchdog_ts) >= state_tolerance_sec:

                            state = 'generateEventCheck'

                            generateEvent_ts = curr_ts

                            print("[FORCED TRANSITION] WATCHDOG → GENERATE EVENT")

                    

                    event_buffer_raw.append(raw_frame)

                    

                    

                elif state == 'generateEventCheck':

                    print("-------- EVENT DETECTED --------")

                    print(f"Watchdog Time: {watchdog_ts}, GenerateEvent Time: {generateEvent_ts}")

                    generate_event_time = datetime.fromtimestamp(generateEvent_ts).strftime("%Y-%m-%d_%H-%M-%S")

                    events.append({

                        "watchdog_time":    datetime.fromtimestamp(watchdog_ts).strftime("%Y-%m-%d_%H-%M-%S"),

                        "generate_event_time":  generate_event_time

                    })

                    state = 'start'

                    last_event_generated_ts = time.time()

                    cv2.imwrite(os.path.join(livefeed_dir, f'{start_time_str}_{generate_event_time}_event.png'), frame)

                    output_event_path = os.path.join(livefeed_dir, f"{start_time_str}_{generate_event_time}_event.mp4")

                    print(f"-------------------- {len(event_buffer_raw)} ------------------------")

                    event_video_queue.put((

                        event_buffer_raw.copy(),  # Or event_buffer_annotated.copy()

                        output_event_path,

                        config.get('analysis_engine', {}).get('fps'),

                        (frame_width, frame_height)

                    ))



                    with open(os.path.join(livefeed_dir, f"{start_time_str}_{generate_event_time}_events.json"), "w") as f:

                        json.dump(events, f, indent=4)





            # Annotations

            state_text = f"STATE: {state.upper()}"

            font = cv2.FONT_HERSHEY_SIMPLEX

            font_scale = 0.8

            font_thickness = 2

            text_size, _ = cv2.getTextSize(state_text, font, font_scale, font_thickness)

            text_x = frame_width - text_size[0] - 10

            text_y = 30



            cv2.putText(frame, state_text, (text_x, text_y), font, font_scale, (0, 255, 255), font_thickness)



            # Draw start time string at bottom-right

            timestamp_text = f"Start: {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

            timestamp_size, _ = cv2.getTextSize(timestamp_text, font, font_scale, font_thickness)

            timestamp_x = frame_width - timestamp_size[0] - 10

            timestamp_y = frame_height - 10  # 10 pixels from bottom



            cv2.putText(frame, timestamp_text, (timestamp_x, timestamp_y), font, font_scale, (255, 255, 0), font_thickness)



            annotated_controlled_fps_writer.write(

                cv2.resize(

                    frame, (window_width, window_height)

                )

            )



            live_feed_controlled_fps_writer.write(

                cv2.resize(

                    raw_frame, (window_width, window_height)

                )

            )



            # Show live stream

            if config.get("video",{}).get('display',{}).get("flag"):

                cv2.imshow(

                    'RTSP Camera Annotated Controlled Feed', 

                    cv2.resize(

                        frame, (window_width, window_height)

                        )

                    )

                if cv2.waitKey(1) & 0xFF == ord('q'):

                    break        



                cv2.imshow(

                    'RTSP Live Camera Controlled Feed', 

                    cv2.resize(

                        raw_frame, (window_width, window_height)

                        )

                    )

                if cv2.waitKey(1) & 0xFF == ord('q'):

                    break



        live_feed_full_fps_writer.write(cv2.resize(raw_frame, (window_width, window_height)))

            

            



    # Finalize

    live_feed_full_fps_writer.release()

    live_feed_controlled_fps_writer.release()

    annotated_controlled_fps_writer.release()

    cap.release()

    with open(INFERENCE_TIMES_FILE, "w") as f:
        json.dump(inference_times, f, indent=2)


    cv2.destroyAllWindows()



    with open(os.path.join(livefeed_dir, f"{start_time_str}_meta_data.json"), "w") as f:

        json.dump(meta_data_list, f, indent=4)



    with open(os.path.join(livefeed_dir, f"{start_time_str}_events.json"), "w") as f:

        json.dump(events, f, indent=4)



    print("\nAll detected events:")

    for e in events:

        print(e)

            



             











