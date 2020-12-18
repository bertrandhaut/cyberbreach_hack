import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from cyberhack.hack import analyze_file


logger = logging.getLogger(__name__)


class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        time.sleep(2)  # wait for the file to be completely written
        file_path = event.src_path

        logger.info(f'Starting analysis of {file_path}')
        analyze_file(file_path)


if __name__ == "__main__":
    logger.info('Cyberhack hack started.')
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
