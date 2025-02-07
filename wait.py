import time
import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

file_sizes = {}  # 최근 변경된 파일 크기 저장
last_executed = {}  # 파일별 마지막 실행 시간 저장
EXECUTION_DELAY = 2  # 중복 실행 방지 대기 시간 (초)

class ImageFileEventHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            time.sleep(0.5)  # 파일 저장이 완료되도록 대기
            file_size = os.path.getsize(event.src_path)

            # 파일 크기가 변하는 동안 대기 (파일이 완전히 저장될 때까지)
            for _ in range(5):  # 최대 2.5초 대기 (0.5초 x 5회)
                new_size = os.path.getsize(event.src_path)
                if new_size == file_size:  
                    break  # 크기가 변하지 않으면 저장 완료로 간주
                file_size = new_size
                time.sleep(0.5)

            # 동일한 크기의 중복 감지 방지
            if event.src_path in file_sizes and file_sizes[event.src_path] == file_size:
                return  

            # 마지막 실행 후 일정 시간 지난 후 실행
            current_time = time.time()
            if event.src_path in last_executed and (current_time - last_executed[event.src_path]) < EXECUTION_DELAY:
                return  

            file_sizes[event.src_path] = file_size  # 새로운 크기 저장
            last_executed[event.src_path] = current_time  # 마지막 실행 시간 저장
            print(f"이미지 파일 변경 감지: {event.src_path}")

            # 변경된 이미지 파일을 처리하는 다른 파이썬 코드 실행
            subprocess.run(["python", "calmarker.py", event.src_path])

if __name__ == "__main__":
    path_to_watch = r"C:/Users/UserK/Desktop/raw"  # 감시할 폴더 경로

    event_handler = ImageFileEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=False)
    observer.start()
    print(f"{path_to_watch} 디렉토리를 감시 중입니다...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
