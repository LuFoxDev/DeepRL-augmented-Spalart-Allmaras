import os
import pickle
import uuid
import shutil

class IOdict():

    def __init__(self, logger, max_size=1000):
        self.max_size = max_size
        self.logger = logger
        self.uuid = uuid.uuid4()
        self.path_for_tempfile = os.path.abspath(os.path.join(f"__iodict_temp_{self.uuid}"))
        self.logger.info(f"initialized IODict, saving temp files to: {self.path_for_tempfile}")
        os.makedirs(self.path_for_tempfile)
        self.in_mem_dict = {}
        self.finder = {}
        
    def get(self, *args):
        if len(args) == 1:
            # in this case only a key is passed
            key = args

            if key not in self.in_mem_dict.keys():
                self.logger.info(f"the key {key} is not in the in-memory dictionary. now dumping current and loading required dict.")
                self.dump_dict()
                self.load_dict(key)

            return self.in_mem_dict[key]

        elif len(args) == 2:
            # in this case the sub dict shall be accessed - a key to the main dict is passed and then the key for the subdict is passed
            key_main, key_sub = args
            if key_main not in self.in_mem_dict.keys():
                self.logger.info(f"the key {key_main} is not in the in-memory dictionary. now dumping current and loading required dict.")
                self.dump_dict()
                self.load_dict(key_main)

            return self.in_mem_dict[key_main][key_sub] 

    def set(self, *args):
        if len(args) == 2:
            # in this case a key and a value is passed
            key, obj = args
            if len(self.in_mem_dict.keys()) <= self.max_size:
                self.in_mem_dict[key] = obj
            else:
                self.logger.info(f"the max number of keys ({self.max_size}) in the in-memory dict is exceeded, now dumping current dict and initializing empty dict.")
                self.dump_dict()
                self.in_mem_dict[key] = obj
        elif len(args) == 3:
            # in this case the sub dict shall be accessed - a key to the main dict is passed and then the key and a value for the subdict is passed
            key_main, key_sub, obj = args
            if len(self.in_mem_dict[key_main].keys()) <= self.max_size:
                self.in_mem_dict[key_main][key_sub] = obj
            else:
                self.logger.info(f"the max number of keys ({self.max_size}) in the in-memory dict is exceeded, now dumping current dict and initializing empty dict.")
                self.dump_dict()
                self.in_mem_dict[key_main] = {}
                self.in_mem_dict[key_main][key_sub] = obj
        else:
            raise ValueError("an invalid number of arguments were passed to the iodict.set method. Either 2 or 3 can be used.")

    def dump_dict(self):
        new_uuid = uuid.uuid4()
        old_files_that_need_to_be_deleted = []
        full_path_to_file = os.path.join(self.path_for_tempfile, str(new_uuid))
        self.logger.info(f"dumping dict: {full_path_to_file}")
        with open(full_path_to_file, 'wb') as handle:
            pickle.dump(self.in_mem_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for key in self.in_mem_dict.keys():
            if key in self.finder.keys():
                # a file for this key exists, the file has to be deleted
                old_files_that_need_to_be_deleted.append(self.finder[key])
            self.finder[key] = full_path_to_file
        del self.in_mem_dict
        # delete old files
        for file in list(set(old_files_that_need_to_be_deleted)):
            try:
                os.remove(file)
            except:
                self.logger.warning("cannot delete file: "+file)
        self.in_mem_dict = {}
   
    def load_dict(self, key):
        self.logger.info(f"opening dict")
        full_path = self.finder[key]
        with open(full_path, 'rb') as handle:
            self.in_mem_dict = pickle.load(handle)

    def close(self):
        self.logger.info("deleting all iodict files")
        size_GB = sum(d.stat().st_size for d in os.scandir(self.path_for_tempfile) if d.is_file())/(1024*1024*1024)
        self.logger.info(f"size of iodict files : {size_GB:.2f} GB")
        shutil.rmtree(self.path_for_tempfile)


if __name__ == "__main__":
    from tqdm import tqdm
    import numpy as np

    iodict = IOdict(max_size=2000)

    rnd_size = 100000
    loop_size = 10000
    # create
    for i in tqdm(range(loop_size)):
        iodict.set(i, {})
        iodict.set(i, "random_matrix", np.random.rand(rnd_size)) 
    
    # access
    for i in tqdm(range(loop_size)):
        temp_var = iodict.get(i, "random_matrix")

    iodict.close()