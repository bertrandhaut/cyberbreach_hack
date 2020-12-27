import logging

import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from cyberbreach.config import configure_logger
from cyberbreach.hack import analyze_file


logger = logging.getLogger(__name__)


class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        time.sleep(2)  # wait for the file to be completely written
        file_path = event.src_path

        logger.info(f'Starting analysis of {file_path}')
        try:
            analyze_file(file_path)
        except Exception as e:
            logger.exception(f'Error dealing with {file_path}')


if __name__ == "__main__":
    configure_logger()
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
