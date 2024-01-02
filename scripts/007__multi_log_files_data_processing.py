import re
from datetime import datetime
from pathlib import Path

from joblib import Parallel, delayed, parallel_config


def parse_log_line(log_line):
    log_pattern = r"\[(?P<datetime>.*?)\] (?P<level>\w+): (?P<message>.*)"
    log_match = re.match(log_pattern, log_line)

    log_datetime = datetime.strptime(log_match.group("datetime"), "%Y-%m-%d %H:%M:%S")
    log_level = log_match.group("level")
    log_message = log_match.group("message")
    return log_datetime, log_level, log_message


def process_log_file(log_file=None):
    with open(log_file, "r") as file:
        log_lines = file.readlines()
        with parallel_config():
            logs = Parallel(n_jobs=-1)(delayed(parse_log_line)(log_line) for log_line in log_lines)
        return logs


def glob_log_files(logs_dir=None):
    logs_dir_path = Path(logs_dir).expanduser().resolve()
    yield from logs_dir_path.glob("*.txt")


log_files = glob_log_files(logs_dir="./data/raw/logs")
with parallel_config():
    logs = Parallel(n_jobs=-1)(delayed(process_log_file)(log_file) for log_file in log_files)

print(logs)
