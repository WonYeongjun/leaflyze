import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

# 전역 변수 설정
last_executed = {}  # 파일별 마지막 실행 시간 저장
EXECUTION_DELAY = 2  # 중복 실행 방지 대기 시간 (초)
is_running = False  # 파이썬 코드 실행 중 여부
lock = threading.Lock()  # 스레드 안전성을 위한 락


class ImageFileEventHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global is_running
        with lock:
            if is_running:
                return

            if not event.is_directory and event.src_path.lower().endswith(
                (".jpg", ".jpeg", ".png", ".gif")
            ):
                time.sleep(0.5)  # 파일 저장이 완료되도록 대기

                current_time = time.time()
                if (
                    event.src_path in last_executed
                    and (current_time - last_executed[event.src_path]) < EXECUTION_DELAY
                ):
                    return
                # 최신 크기 저장 & 실행 시간 갱신
                last_executed[event.src_path] = current_time
                print(f"이미지 파일 변경 감지: {event.src_path}")

                # 변경된 이미지 파일을 처리하는 다른 파이썬 코드 실행
                is_running = True

        # subprocess.run을 비동기적으로 실행
        threading.Thread(target=self.run_subprocess, args=(event.src_path,)).start()

    def run_subprocess(self, src_path):
        global is_running
        subprocess.run(["python", "calibration_marker_detector.py", src_path])
        with lock:
            is_running = False


if __name__ == "__main__":
    path_to_watch = r"C:/Users/UserK/Desktop/raw"  # 감시할 폴더 경로

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
