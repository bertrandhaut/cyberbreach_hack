import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
		time.sleep(2)  # wait for the file to be completely written
		file_path = event.src_path
		
		parse_file(file_path)
		


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    path = r'C:\data\tmp\a\Cyberpunk 2077'
    my_event_handler = MyHandler()
	
    observer = Observer()
    observer.schedule(my_event_handler, path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()