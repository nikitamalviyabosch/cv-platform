import json
from pathlib import Path

### details are accessed from json file
class Common:
    def load_json(self, json_path):
        with open(json_path) as json_file:
            return json.load(json_file)

    def create_directories(self, *dirs):
        for directory in dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
