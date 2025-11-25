from helpers import is_height_greater_than_width, is_width_greater_than_height


class AnalysisEngine():
    def __init__(self, prev_frame_statu, curr_frame_status, ):
        

def ae():
    if prev_frame_status is False and curr_frame_status is True:
        is_width_and_height_line_cut_each_other = True
        start_ts = time.time()

    print(f"prev: {prev_frame_status}, curr: {curr_frame_status}, intersected: {is_width_and_height_line_cut_each_other}")
    meta_data_list.append(metadata)

    # === State Machine ===
    if state == 'start':
        if is_width_and_height_line_cut_each_other:
            state = 'watchdog'
            watchdog_ts = time.time()
            print("[STATE TRANSITION] START → WATCHDOG")
            is_width_and_height_line_cut_each_other = False

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

    elif state == 'generateEventCheck':
        print("-------- EVENT DETECTED --------")
        print(f"Watchdog Time: {watchdog_ts}, GenerateEvent Time: {generateEvent_ts}")
        events.append({
            "watchdog_time":    datetime.fromtimestamp(watchdog_ts).strftime("%Y-%m-%d_%H-%M-%S"),
            "generate_event_time":  datetime.fromtimestamp(generateEvent_ts).strftime("%Y-%m-%d_%H-%M-%S")
        })
        state = 'start'
