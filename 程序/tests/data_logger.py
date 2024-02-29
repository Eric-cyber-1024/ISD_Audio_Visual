import threading
import datetime

class DataLogger:
    def __init__(self, log_interval, file_path):
        self.log_interval = log_interval
        self.data = []
        self.data_lock = threading.Lock()
        self.thread = threading.Thread(target=self._log_data_thread)
        self.is_logging = False
        self.file_path = file_path

    def start_logging(self):
        if not self.is_logging:
            self.is_logging = True
            self.thread.start()

    def stop_logging(self):
        self.is_logging = False
        self.thread.join()

    def _log_data_thread(self):
        while self.is_logging:
            # Write the data to the file
            self.write_to_file()

            # Wait for the specified log interval
            threading.Event().wait(self.log_interval)

    def add_data(self, data):
        with self.data_lock:
            self.data.append(data)

    def write_to_file(self):
        with self.data_lock:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_entries = [f"{timestamp}: {data}\n" for data in self.data]

            with open(self.file_path, "a") as file:
                file.writelines(log_entries)

            self.data.clear()