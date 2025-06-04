import psutil
import time

def find_exam_window_pid(window_title):
    """Find the PID of the process with the given window title (assumes Tkinter app title)."""
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if window_title.lower() in ' '.join(process.info['cmdline']).lower():
                return process.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def monitor_processes(window_title):
    print("Monitoring processes...")

    exam_pid = find_exam_window_pid(window_title)
    if not exam_pid:
        print("Could not find the exam window process.")
        return

    print(f"Exam window PID: {exam_pid}")

    try:
        while True:
            processes = psutil.process_iter(['pid', 'name'])
            flagged_processes = []

            for process in processes:
                try:
                    pid = process.info['pid']
                    name = process.info['name']

                    if pid != exam_pid:
                        flagged_processes.append(process.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            
            if flagged_processes:
                print("Found unrelated processes:")
                for flagged_process in flagged_processes:
                    print(f"PID: {flagged_process['pid']}, Process Name: {flagged_process['name']}")

            time.sleep(5)

    except KeyboardInterrupt:
        print("Exiting process monitoring...")

if __name__ == "__main__":
    monitor_processes(window_title="Exam Window")  