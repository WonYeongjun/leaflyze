import time
import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 최근 변경된 파일 크기를 저장하는 딕셔너리
file_sizes = {}


class ImageFileEventHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith(
            (".jpg", ".jpeg", ".png", ".gif")
        ):
            time.sleep(0.5)  # 잠깐 대기 (쓰기 완료 대기)
            file_size = os.path.getsize(event.src_path)

            # 이전 크기와 비교하여 중복 감지 방지
            if event.src_path in file_sizes and file_sizes[event.src_path] == file_size:
                return  # 동일한 크기면 무시 (즉, 두 번째 감지일 가능성이 높음)

            file_sizes[event.src_path] = file_size  # 새로운 크기 저장
            print(f"이미지 파일 변경 감지: {event.src_path}")

            # 변경된 이미지 파일을 처리하는 다른 파이썬 코드 실행
            subprocess.run(["python", "calmarker.py", event.src_path])


if __name__ == "__main__":
    path_to_watch = r"C:/Users/UserK/Desktop/raw"  # 감시할 폴더 경로 지정

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
