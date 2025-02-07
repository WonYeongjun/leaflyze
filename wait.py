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

            # 파일 크기가 안정될 때까지 대기 (최대 2.5초)
            for _ in range(5):  
                new_size = os.path.getsize(event.src_path)
                if new_size == file_size:  
                    break  # 크기가 변하지 않으면 저장 완료로 간주
                file_size = new_size
                time.sleep(0.5)

            # *** 추가 대기 후 마지막 크기 체크 ***
            time.sleep(1)  
            final_size = os.path.getsize(event.src_path)

            if final_size != file_size:
                return  # 여전히 크기가 변하면 실행 안 함

            # 중복 실행 방지: 마지막 실행 이후 일정 시간 지나야 실행
            current_time = time.time()
            if event.src_path in last_executed and (current_time - last_executed[event.src_path]) < EXECUTION_DELAY:
                return  

            # 최신 크기 저장 & 실행 시간 갱신
            file_sizes[event.src_path] = final_size
            last_executed[event.src_path] = current_time
            print(f"이미지 파일 변경 감지: {event.src_path}")

            # *** 3초 대기 후 실행 ***
            time.sleep(3)

            # 변경된 이미지 파일을 처리하는 다른 파이썬 코드 실행
            subprocess.run(["python", "calmarker_new.py", event.src_path])

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
