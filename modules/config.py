# enable type hinting of the 'Config' class
#from __future__ import annotations
from typing import Any, List, Dict, Union

# python module imports
import configparser
import copy
import os
import json



class Config:
    """
    Class for managing the project config file.
    Reduces normal get and set methods of 'configparser'(config['section']['setting']) to config['setting'].
    This means that setting names must be unique and cannot exist twice, even if in multiple sections!
    """

    def __init__(self) -> None:
        """
        Initializes object and optionally shallow copies an existing 'Config' object.

        """
        # initialize private _raw_config member
        self._raw_config = None

        # define list of config options that shall always be converted to a string
        self._always_convert_to_string = []

        self._raw_config = configparser.RawConfigParser()
        self._filetype = None

    def load(self, path: str) -> None:
        """
        Reads the config file at 'path'.

        path: Path and name to a config file
        """

        # read raw config
        if ".conf" in path:
            self._filetype = "conf"
            self._raw_config.read(path, encoding="utf-8")
        elif ".json" in path:
            # Opening JSON file
            self._filetype = "json"
            with open(path) as json_file:
                self._raw_config =  json.load(json_file)
        else:
            raise ValueError(f"unsupported config file: {path}")


        # check for duplicate settings (in mutiple sections)
        if  self._filetype == "conf":
            settings = set()
            for section in self._raw_config.sections():
                for setting in self._raw_config.options(section):
                    if setting not in settings:
                        settings.add(setting)
                    else:
                        raise ValueError(f"Setting with same name found in multiple sections in config. There cannot be duplicate setting names.")

    def _get_dict(self) -> Dict:
        """
        Returns a dictionary of the config.
        """
        if self._filetype == "conf":
            config_dict = {}
            for section in self._raw_config.sections():
                for setting in self._raw_config.options(section):
                    value = self._raw_config.get(section, setting)
                    if setting in self._always_convert_to_string:
                        config_dict[setting] = str(value)
                    else:
                        config_dict[setting] = self._convert_string(value)
        else:
            config_dict = self._raw_config


        # return config dictionary
        return config_dict

    def save(self, path: str) -> None:
        """
        Saves the 'Config' instance as a config file.

        path: where the config file should be stored & the name of the config file
        """
        # check if path contains invalid charcters
        invalid_characters = ["/", "\\"]
        invalid_characters.remove(os.sep)
        for character in invalid_characters:
            if character in path:
                raise ValueError(f"Invalid character '{character}' found in 'path'.")

        # if path does not exist, create it
        folder_path = path.rpartition(os.sep)[0]
        if not folder_path == "" and not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # add file ending of none is present
        if not ".conf" in path:
            #path = path.strip(".")
            path += ".conf"

        if self._filetype == "conf":
            # save config under name at path
            with open(path, "w", encoding="utf-8") as config_file:
                self._raw_config.write(config_file)
            config_file.close()
        else:
            with open(path, 'w') as fp:
                json.dump(self._raw_config, fp)
        

    def _convert_string(self, string: str) -> Union[int, bool]:
        """
        Converts a string into numbers or booleans respectivly.
        """
        # check if the string is a number
        if string.isnumeric():
            return int(string)
        # check if string is boolean
        if string.lower() == "true":
            return True
        if string.lower() == "false":
            return False
        # if neither number nor boolean
        # return string
        return string

    def copy(self):
        """
        Returns a copy of itself.
        """

        return Config(self)

    # operator overloading
    def __str__(self) -> str:
        # return the config as a dictionary
        return str(self._get_dict())

    def __getitem__(self, key: str) -> Union[bool, int, str]:
        # error checking key type
        if type(key) != str:
            raise TypeError("'key' must be of type 'str'.")

        # error checking if key is present
        config_dict = self._get_dict()
        if key.lower() not in config_dict:
            raise KeyError(f"'{key.lower()}' not found in config.")

        # return value
        return config_dict[key.lower()]

    def __setitem__(self, key: str, value: Union[bool, int, str]) -> None:
        # error checking key and value types
        if type(key) != str:
            raise TypeError("'key' must be of type 'str'.")
        if type(value) != bool and type(value) != int and type(value) != str:
            raise TypeError("'value' must be of type 'bool', 'int' or 'str'")

        if self._filetype == "conf":
            # set value
            key_found = False
            for section in self._raw_config.sections():
                if key in self._raw_config.options(section):
                    self._raw_config[section][key] = value
                    key_found = True

            if not key_found:
                self._raw_config[self._raw_config.sections()[0]][key] = value
        else:
            self._raw_config[key] = value


    def __delitem__(self, key: str) -> None:
        # error checking key type
        if type(key) != str:
            raise TypeError("'key' must be of type 'str'.")
        
        if self._filetype == "conf":
            # set value
            key_found = False
            for section in self._raw_config.sections():
                if key in self._raw_config.options(section):
                    del self._raw_config[section][key]

                    key_found = True

            # error checking if key is present
            if not key_found:
                raise KeyError("'key' not found in config.")
        else:
            self._raw_config.pop(key, None)
