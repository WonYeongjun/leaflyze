import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import json
import os

last_executed = {}
EXECUTION_DELAY = 2
is_running = False
lock = threading.Lock()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)


class ImageFileEventHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global is_running
        with lock:
            if is_running:
                return

            if event.src_path.lower().endswith((".jpg", ".jpeg", ".png")):
                time.sleep(0.5)
                current_time = time.time()
                if (
                    event.src_path in last_executed
                    and (current_time - last_executed[event.src_path]) < EXECUTION_DELAY
                ):
                    return
                last_executed[event.src_path] = current_time
                print(f"이미지 파일 변경 감지: {event.src_path}")

                is_running = True

        threading.Thread(target=self.run_subprocess, args=(event.src_path,)).start()

    def run_subprocess(self, src_path):
        global is_running
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), config["main_code_name"]
        )
        subprocess.run(["python", script_path, src_path])
        with lock:
            is_running = False


if __name__ == "__main__":
    path_to_watch = config["pc_watching_folder"]
    # path_to_watch = "C:/Users/UserK/Desktop/raw/"
    observer = Observer()
    event_handler = ImageFileEventHandler()
    observer.schedule(event_handler, path_to_watch, recursive=False)
    observer.start()
    print(f"{path_to_watch} 디렉토리를 감시 중입니다...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
