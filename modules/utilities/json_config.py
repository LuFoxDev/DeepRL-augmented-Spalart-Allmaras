
import json

class json_config():

    def __init__(self, path_to_config_file):
        self.path_to_config_file = path_to_config_file

        with open(self.path_to_config_file) as config_file:
            self.config = json.load(config_file)


    def __getattr__(self, attr):
        return self.config[attr]

