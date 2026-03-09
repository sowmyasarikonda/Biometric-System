import csv
import time
import psutil
import os

class MetricsLogger:
    def __init__(self, filename="metrics_log.csv"):

        self.filename = filename

        if not os.path.exists(self.filename):
            with open(self.filename, "w", newline="") as f:
                writer = csv.writer(f)

                writer.writerow([
                    "timestamp",
                    "identity",
                    "similarity_score",
                    "match",
                    "latency_ms",
                    "cpu_percent",
                    "memory_percent"
                ])

    def log(self, identity, similarity, match, latency):

        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent

        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                time.time(),
                identity,
                round(similarity,4),
                match,
                round(latency,2),
                cpu,
                memory
            ])