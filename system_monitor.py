import psutil
import time
import logging
from datetime import datetime
import json
import os

# Set up logging
logging.basicConfig(filename='system_monitor.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Define a class for the system monitor
class SystemMonitor:
    def __init__(self, interval=5, log_file='system_usage.json'):
        self.interval = interval
        self.log_file = log_file
        self.data = []

    def record_usage(self):
        cpu_usage = psutil.cpu_percent(interval=self.interval)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        timestamp = datetime.now().isoformat()

        usage_data = {
            'timestamp': timestamp,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'disk_usage': disk_usage
        }
        self.data.append(usage_data)
        logging.info(f'Recorded usage: {usage_data}')

    def save_to_json(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=4)
        logging.info(f'Saved data to {self.log_file}')

    def start_monitoring(self):
        try:
            while True:
                self.record_usage()
                time.sleep(self.interval)
        except KeyboardInterrupt:
            logging.info('Monitoring stopped by user.')
            self.save_to_json()

# Entry point of the program
if __name__ == '__main__':
    monitor = SystemMonitor(interval=5)
    monitor.start_monitoring()