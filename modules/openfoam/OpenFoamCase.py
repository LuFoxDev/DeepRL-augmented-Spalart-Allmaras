

from logging import Logger
from modules.config import Config
from modules.openfoam.read_OF_coordinates import read_OF_internal_field_coordinates, read_OF_inlet_coordinates, read_OF_internal_field_results
from modules.plotting.plotting_functions import plot_coordinates, plot_scalar_field_on_mesh, plot_interpolated_cell_batch, plot_cell_batch_coverage, plot_interpolation_as_scatterplot, plot_interpolation_new
from modules.utilities.terminal import run_command
from modules.utilities.file_helpers import modify_file
from modules.utilities.units import get_unit
from modules.utilities.iodict import IOdict
from tqdm import tqdm
import os
import re
import glob
import numpy as np
from shutil import copyfile, copytree, move, rmtree
from pathlib import Path
import stat
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial.qhull import QhullError
import time
import pickle
import pandas as pd

class OpenFoamCase():

    def __init__(self, logger : Logger, config : Config) -> None:
        
        self.logger = logger
        self.config = config
        
        self.path_to_export_folder_for_this_run = os.path.join(
            self.config["export_folder"], self.config["name_of_run"])

        self.path_to_openfoam_case_folder = os.path.join(
            self.config["export_folder"], self.config["name_of_run"], "OpenFoamCase", "")

        self.path_to_postprocessing_folder = os.path.join(
            self.config["export_folder"], self.config["name_of_run"], "post-processing", "")

        self.path_to_openfoam_templates = os.path.join(
            "./templates", "openfoam", self.config["name_of_template"])

        self.path_to_openfoam_case_template = os.path.join(
            "./templates", "openfoam", self.config["name_of_template"], "Case")


        self.error_coordinate_limits = [[float(self.config["x_min"]), float(self.config["x_max"])],
                                        [float(self.config["y_min"]), float(self.config["y_max"])]]

        self.cell_batches_plotted = False
        self.interpolation_method_for_cell_batches = self.config["interpolation_method"]
        self.subsampling_ratio = self.config["subsampling_ratio"]
        self.time_list = None
        self.results_by_time = None
        self.absolute_error_by_dir_by_time = []
        self.error_by_time = []
        self.rmse_by_time = []
        self.area_weighted_rmse_by_time = []
        self.area_weighted_error_by_time = []
        # self.cell_dict = {}
        self.cell_dict = IOdict(logger=self.logger, max_size=1000)
        self.epochs_per_episode = self.config["epochs_per_episode"]
        self.min_rmse = 1.0
        self.time_of_min_rmse = 1

        self.get_unit = get_unit
        
        pass

    def setup_new_run(self, delete_old_files=False) -> None:
        """
        Set up a new clean run: 
        - create new folder
        - copy template
        - create mesh
        - extract cell center coordinates
        """
        self.logger.info("setting up new run")
        self.run_log = ""
        self.mesh_log = ""
        if delete_old_files and os.path.exists(self.path_to_export_folder_for_this_run):
            self.logger.info("deleting old existing exports")
            rmtree(self.path_to_export_folder_for_this_run)
        self._create_new_folder()
        self._copy_template()
        self._create_BlockMesh()
        if self.config["parallel_run"]:
            # copy decomposeParDict to system folder
            self._copy_decomposeParDict()
        self._extractCellCenterCoordinates()

        # safe used config
        self.config.save(os.path.join(self.config["export_folder"], self.config["name_of_run"], "used_config.conf"))

    def _copy_decomposeParDict(self):
        """
        open template decomposeParDict and add simple decomposion settings:
        numberOfSubdomains, simpleCoeffs_n_x, simpleCoeffs_n_y
        """
        decomposition_mode = self.config["decomposition_mode"]
        _source = os.path.join(self.path_to_openfoam_templates, f"decomposeParDict_{decomposition_mode}")
        _destination = os.path.join(self.path_to_openfoam_case_folder, "system", "decomposeParDict")

        # modify decomposeParDict
        placeholders = ["#numberOfSubdomains#", "#simpleCoeffs_n_x#", "#simpleCoeffs_n_y#"]

        numberOfSubdomains = self.config["numberofsubdomains"]
        n_y = 1 if numberOfSubdomains % 2 !=0  else 2
        n_x = int(numberOfSubdomains/n_y)
        # test
        assert n_x *  n_y == numberOfSubdomains

        replacements = [numberOfSubdomains, n_x, n_y]

        modify_file(_source, _destination, placeholders, replacements)

        with open(_source, "r") as f_source:
            decomposeParDictFile = f_source.read()
        f_source.close()

        with open(_destination, "w") as f:
            for placeholder, replacement in zip(placeholders, replacements):
                decomposeParDictFile = decomposeParDictFile.replace(placeholder, str(replacement))
            f.write(decomposeParDictFile)
        f.close()
        self.logger.info("created decomposeParDict file for parallel run")




    def _create_new_folder(self) -> None:
        """
        creates new empty folder in desired export folder
        """

        _subfolders = [self.path_to_openfoam_case_folder, self.path_to_postprocessing_folder]

        for _folder in _subfolders:
            if not os.path.exists(_folder):
                os.makedirs(_folder)
                self.logger.info(f"created folder: {_folder}")
            else:
                self.logger.info(f"export folder: {_folder} already exists")

    def _copy_template(self) -> None:
        """
        copy template for openFoam case to export folder
        """
        _source = self.path_to_openfoam_case_template
        _destination = self.path_to_openfoam_case_folder
        copytree(_source, _destination, dirs_exist_ok=True)
        self.logger.info(f"copied template {self.config['name_of_template']}")


    def _create_BlockMesh(self) -> None:
        """
        execute BlockMesh command
        """
        _source = "./scripts/create_BlockMesh.sh"
        _target = os.path.join(self.path_to_openfoam_case_folder, "create_BlockMesh.sh")
        path_to_script = copyfile(_source, _target)

        f = Path(path_to_script)
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
        command = f"bash {path_to_script}"
        self.mesh_log += run_command(command)
        
        with open(os.path.join(self.path_to_export_folder_for_this_run,"blockMesh_output.txt"), "w") as text_file:
            text_file.write(self.mesh_log)
            
    def _extractCellCenterCoordinates(self) -> None: 
        """
        runs "writeCellCentres" to extract domain coordinates
        """
        self.logger.info("now extracting cell center coordinates")
        # run writeCellCentres
        _source = "./scripts/writeCellCentres.sh"
        _target = os.path.join(self.path_to_openfoam_case_folder, "writeCellCentres.sh")
        path_to_script = copyfile(_source, _target)
        f = Path(path_to_script)
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
        command = f"bash {path_to_script}"
        results = run_command(command)
        with open(os.path.join(self.path_to_export_folder_for_this_run,"witeCellCentres_output.txt"), "w") as text_file:
            text_file.write(results)

        # move ccx, ccy, ccz to postprocessing
        cellCentreFilenames = ["ccx", "ccy", "ccz"]
        target_folder_for_cellCentreFiles = os.path.join(self.path_to_postprocessing_folder)
        for _f in cellCentreFilenames:
            _source = os.path.join(self.path_to_openfoam_case_folder, "0", _f)
            _target = os.path.join(target_folder_for_cellCentreFiles, _f)
            move(_source, _target)

        # import coordinates
        #######################################################################
        self.logger.info("read blockmesh coordinates")
        self.BlockMesh_coordinates = read_OF_internal_field_coordinates(target_folder_for_cellCentreFiles)
        self.BlockMesh_coordinates_2D = self.BlockMesh_coordinates[:,[0,1]]

        self._extract_cell_surface_areas()

        if self.config["make_plots"]:
            plot_coordinates(self.BlockMesh_coordinates[:, 0], self.BlockMesh_coordinates[:, 1],
                            self.BlockMesh_coordinates[:, 2], source="own mesh_non vtk",
                            export_path=self.path_to_postprocessing_folder, show_all=True)
    
    def _extract_cell_surface_areas(self):
        self.logger.info("extracing cell surface area")
        self.cell_areas = np.zeros_like(self.BlockMesh_coordinates_2D[:,0])

        for cell_id in tqdm(range(self.BlockMesh_coordinates_2D.shape[0])):

            cell_x = self.BlockMesh_coordinates_2D[cell_id, 0]
            cell_y = self.BlockMesh_coordinates_2D[cell_id, 1]

            _all_others_x = self.BlockMesh_coordinates_2D[np.where(self.BlockMesh_coordinates_2D[:, 0] != cell_x), 0]
            _all_others_y = self.BlockMesh_coordinates_2D[np.where(self.BlockMesh_coordinates_2D[:, 1] != cell_y), 1]
            min_dist_x = np.abs(np.min(_all_others_x-cell_x))
            min_dist_y = np.abs(np.min(_all_others_y-cell_y))
            area = min_dist_x * min_dist_y
            self.cell_areas[cell_id] = area

        self.cell_areas_normalized = self.cell_areas / np.mean(self.cell_areas)

        plot_scalar_field_on_mesh(self.BlockMesh_coordinates_2D[:,0], self.BlockMesh_coordinates_2D[:,1],
                            self.cell_areas_normalized,
                            f"normalized cell area",
                            "x", "y", "1",  f"normalized_cell_area",
                            self.path_to_postprocessing_folder,
                            close_up_limits=None, close_up_aspect_ratio=2, cmap="jet", center_cbar=False, aspect_ratio=None)


    def get_inlet_boundary_coodinates(self) -> np.array:
        """
        extract the coordinates of the inlet boundary
        """
        sim_inlet_coordinates = read_OF_inlet_coordinates(self.path_to_postprocessing_folder)

    def set_inlet_boundary_condition(self, inlet_conditions) -> None:
        """

        """
        template_folder = self.path_to_openfoam_templates
        target_folder =   os.path.join(self.path_to_openfoam_case_folder, "0")
        path_to_template_file = "U_template"
        name_of_new_file = "U"

        # set z-coordinate to zero!
        inlet_conditions[2] = np.zeros(inlet_conditions[2].shape)

        with open(os.path.join(template_folder, path_to_template_file)) as f:
            #lines = f.readlines()
            complete_text = f.read()
            counter = inlet_conditions[0].shape[0]
            vectors_text = ""
            for i in range(counter):
                vectors_text += f"(   {inlet_conditions[0][i]:.4f}   {inlet_conditions[1][i]:.4f}  {inlet_conditions[2][i]:.4f})\n"

            complete_text = complete_text.replace("#counter", str(counter))
            complete_text = complete_text.replace("#vectors", vectors_text)

        with open(os.path.join(target_folder, name_of_new_file), 'w') as f:
            f.write(complete_text)

    def copy_results_folder(self, source_time: int, target_time: int):
        """
        copies content of folder "1000" to folder "1" for exmaple if source_time = 1000, target_time = 1
        """

        _source = os.path.join(self.path_to_openfoam_case_template, str(source_time))
        _destination =  os.path.join(self.path_to_openfoam_case_folder, str(target_time))
        copytree(_source, _destination, dirs_exist_ok=True)
        self.logger.info(f"copied template {self.config['name_of_template']}")




    def set_inlet_and_fuel_injector_boundary_condition(self, inlet_conditions, fuel_injector_conditions) -> None:
        """

        """

        template_folder = self.path_to_openfoam_templates
        target_folder =   os.path.join(self.path_to_openfoam_case_folder, "0")
        path_to_template_file = "U_template"
        name_of_new_file = "U"

        # set z-coordinate to zero!
        fuel_injector_conditions[2] = np.zeros(fuel_injector_conditions[2].shape)
        # fuel_injector_conditions[1] = np.zeros(fuel_injector_conditions[1].shape)
        # fuel_injector_conditions[0] = 0.1* np.ones(fuel_injector_conditions[0].shape)
        inlet_conditions[2] = np.zeros(inlet_conditions[2].shape)

        with open(os.path.join(template_folder, path_to_template_file)) as f:
            complete_text = f.read()

            # FUEL INJECTOR
            counter = fuel_injector_conditions[0].shape[0]
            vectors_text = ""
            for i in range(counter):
                vectors_text += f"(   {fuel_injector_conditions[0][i]:.4f}   {fuel_injector_conditions[1][i]:.4f}  {fuel_injector_conditions[2][i]:.4f})\n"
            
            complete_text = complete_text.replace("#counter_fuelInjector", str(counter))
            complete_text = complete_text.replace("#vectors_fuelInjector", vectors_text)

            # INLET
            counter = inlet_conditions[0].shape[0]
            vectors_text = ""
            for i in range(counter):
                vectors_text += f"(   {inlet_conditions[0][i]:.4f}   {inlet_conditions[1][i]:.4f}  {inlet_conditions[2][i]:.4f})\n"

            complete_text = complete_text.replace("#counter", str(counter))
            complete_text = complete_text.replace("#vectors", vectors_text)

        with open(os.path.join(target_folder, name_of_new_file), 'w') as f:
            f.write(complete_text)        

    def run(self, **controlDict_settings) -> None:
        """
        run simulation (decompose, run solver and reconstruct)
        """

        # change control dict if controlDict_settings are passed

        path_to_controlDict = os.path.join(self.path_to_openfoam_case_folder, "system", "controlDict")
        # read current controlDict file
        with open(path_to_controlDict, mode="r") as f:
            controlDict_input_lines = f.readlines() # f.read()
        # replace values
        controlDict_output_text = ""
        changes_made = False
        for line in controlDict_input_lines:
            new_line = line
            for key in controlDict_settings.keys():
                if line.startswith(key):
                    new_line = f"{key}       {controlDict_settings[key]};\n"
                    changes_made = True
                    self.logger.info(f"changed {key} in controlDict to {controlDict_settings[key]}")

            controlDict_output_text += new_line
        # save new file, if changes were made
        if changes_made:
            with open(path_to_controlDict, mode="w") as f:
                f.write(controlDict_output_text)

            self.logger.info("saved new controlDict")

        # execute bash script to start simulation
        self.logger.info(f"now running {self.config['solver_name']}")
        if self.config["parallel_run"]:
            # for parallel run change numberOfSubdomains in script template
            _source = "./scripts/run_solver_in_parallel.sh"
            _target = os.path.join(self.path_to_openfoam_case_folder, "run_solver_in_parallel.sh")
            with open(_source, mode="r") as f:
                script_text = f.read()
            with open(_target, mode="w") as f:
                script_text = script_text.replace("#numberOfSubdomains#", str(self.config["numberofsubdomains"]))
                script_text = script_text.replace("#solver#", str(self.config["solver_name"]))
                f.write(script_text)
            path_to_script = _target
        
        else:
            # for non parallel just copy the script
            _source = "./scripts/run_solver.sh"
            _target = os.path.join(self.path_to_openfoam_case_folder, "run_solver.sh")
            with open(_source, mode="r") as f:
                script_text = f.read()
            with open(_target, mode="w") as f:
                script_text = script_text.replace("#solver#", str(self.config["solver_name"]))
                f.write(script_text)
            path_to_script = _target
        
        f = Path(path_to_script)
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
        command = f"bash {path_to_script}"
        output_current_run = run_command(command)
        self.run_log += output_current_run

        if output_current_run is None:
            self.last_run_failed = True
        else:
            self.last_run_failed = False
        #self.last_run_failed = False
        self.path_to_exported_log_file = os.path.join(self.path_to_export_folder_for_this_run,"run_log.txt")
        
        # with open(self.path_to_exported_log_file, "w") as text_file:
        #     text_file.write(self.run_log)


    def decompose(self) -> None:
        """
        run decomposition script
        """

        # execute bash script to start simulation
        self.logger.info(f"now running decomposition")
        if self.config["parallel_run"]:
            # for parallel run change numberOfSubdomains in script template
            _source = "./scripts/decompose.sh"
            _target = os.path.join(self.path_to_openfoam_case_folder, "decompose.sh")
            with open(_source, mode="r") as f:
                script_text = f.read()
            with open(_target, mode="w") as f:
                f.write(script_text)
            path_to_script = _target

            f = Path(path_to_script)
            f.chmod(f.stat().st_mode | stat.S_IEXEC)
            command = f"bash {path_to_script}"
            results = run_command(command)

            _log_file = os.path.join(self.path_to_export_folder_for_this_run,"decompose_log.txt")
            
            with open(_log_file, "w") as text_file:
                text_file.write(results)

        else:
            self.logger.warning("decomposition script can only be executed in a parallel run!")
        


    def reconstruct(self) -> None:
        """
        run reconstruction script
        """

        # execute bash script to start simulation
        self.logger.info(f"now running reconstruction")
        if self.config["parallel_run"]:
            # for parallel run change numberOfSubdomains in script template
            _source = "./scripts/reconstruct.sh"
            _target = os.path.join(self.path_to_openfoam_case_folder, "reconstruct.sh")
            with open(_source, mode="r") as f:
                script_text = f.read()
            with open(_target, mode="w") as f:
                f.write(script_text)
            path_to_script = _target

            f = Path(path_to_script)
            f.chmod(f.stat().st_mode | stat.S_IEXEC)
            command = f"bash {path_to_script}"
            results = run_command(command)

            _log_file = os.path.join(self.path_to_export_folder_for_this_run,"reconstruct_log.txt")
            
            with open(_log_file, "w") as text_file:
                text_file.write(results)
        else:
            self.logger.warning("reconstruct script can only be executed in a parallel run!")
        

    def run_solver(self, **controlDict_settings) -> None:
        """
        run only the solver. (decomposition has to be run firsts)
        """

        _start = time.time()
        # change control dict if controlDict_settings are passed
        path_to_controlDict = os.path.join(self.path_to_openfoam_case_folder, "system", "controlDict")
        # read current controlDict file
        with open(path_to_controlDict, mode="r") as f:
            controlDict_input_lines = f.readlines() # f.read()
        # replace values
        controlDict_output_text = ""
        changes_made = False
        for line in controlDict_input_lines:
            new_line = line
            for key in controlDict_settings.keys():
                if line.startswith(key):
                    new_line = f"{key}       {controlDict_settings[key]};\n"
                    changes_made = True
                    self.logger.info(f"changed {key} in controlDict to {controlDict_settings[key]}")

            controlDict_output_text += new_line
        # save new file, if changes were made
        if changes_made:
            with open(path_to_controlDict, mode="w") as f:
                f.write(controlDict_output_text)

            self.logger.info("saved new controlDict")

        # execute bash script to start simulation
        self.logger.info(f"now running {self.config['solver_name']}")
        if self.config["parallel_run"]:
            # for parallel run change numberOfSubdomains in script template
            _source = "./scripts/run_only_solver_in_parallel.sh"
            _target = os.path.join(self.path_to_openfoam_case_folder, "run_only_solver_in_parallel.sh")
            with open(_source, mode="r") as f:
                script_text = f.read()
            with open(_target, mode="w") as f:
                script_text = script_text.replace("#numberOfSubdomains#", str(self.config["numberofsubdomains"]))
                script_text = script_text.replace("#solver#", str(self.config["solver_name"]))
                f.write(script_text)
            path_to_script = _target
        
        else:
            # for non parallel just copy the script
            _source = "./scripts/run_solver.sh"
            _target = os.path.join(self.path_to_openfoam_case_folder, "run_solver.sh")
            with open(_source, mode="r") as f:
                script_text = f.read()
            with open(_target, mode="w") as f:
                script_text = script_text.replace("#solver#", str(self.config["solver_name"]))
                f.write(script_text)
            path_to_script = _target
        
        f = Path(path_to_script)
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
        command = f"bash {path_to_script}"
        output_current_run = run_command(command)
        self.run_log += output_current_run

        if output_current_run is None:
            self.last_run_failed = True
        else:
            self.last_run_failed = False

        self.path_to_exported_log_file = os.path.join(self.path_to_export_folder_for_this_run,"run_log.txt")
        
        with open(self.path_to_exported_log_file, "w") as text_file:
            text_file.write(self.run_log)
        
        _end = time.time()
        self.logger.info(f"TIMER: running solver took {(_end-_start)/60:.1f} minutes")

    def import_results(self, manual_path_to_openfoam_case_folder : str = None) -> None:
        """
        read all reasults from openFoam run
        """
        if manual_path_to_openfoam_case_folder is not None:
            self.path_to_openfoam_case_folder = manual_path_to_openfoam_case_folder
        all_elements_in_OF_folder = glob.glob(self.path_to_openfoam_case_folder + "*")
        results_folder = [folder for folder in all_elements_in_OF_folder if os.path.isfile(folder + "/U") and folder[-2:] != '/0']
        self.time_list_for_current_import = [int(f.split("/")[-1]) for f in results_folder]
        self.time_list_for_current_import.sort()
        if self.time_list is None and self.results_by_time is None:
            self.time_list = self.time_list_for_current_import
            self.results_by_time = {}
        else:
            # remove already imported results from time_list_for_import
            self.time_list_for_current_import = [t for t in self.time_list_for_current_import if t not in self.results_by_time.keys()]
            self.time_list += self.time_list_for_current_import

        for time in self.time_list_for_current_import:
            self.logger.info(f"importing results for t={time}")
            self.results_by_time[time] = {}
            path_to_simulation_results_by_time = f'{self.path_to_openfoam_case_folder}{time}/'
            for param in [f for f in os.listdir(path_to_simulation_results_by_time) if os.path.isfile(path_to_simulation_results_by_time + f)]:
                self.results_by_time[time][param] = read_OF_internal_field_results(param, path_to_simulation_results_by_time)
            
            #simulation_results = read_OF_internal_field_results("U", path_to_simulation_results_by_time)
            #self.results_by_time[time] = simulation_results

        self.results = self.results_by_time[self.time_list[-1]]

    def import_log(self, manual_path_to_logfile : str = None) -> None:
        """
        read log file and plot ressiduals
        """

        if manual_path_to_logfile:
            path_to_logfile = manual_path_to_logfile
        else:
            path_to_logfile = self.path_to_exported_log_file

        with open(path_to_logfile, "r" ) as logfile:
            fullstring = logfile.read()
            #lines = logfile.readlines()

        lines = fullstring.split("\n")

        # find paramters
        parameters = []
        for line in lines:
            regsearch = re.search(r"Solving for ([a-zA-Z]+)", line)
            if regsearch:
                parameter = regsearch.group(1)
                if parameter not in parameters:
                    parameters.append(parameter)

        # extract only part of log of run 
        _fullstring = fullstring
        run_string = ""
        while _fullstring.find("Starting time loop") != -1:
            start_index = _fullstring.find("Starting time loop")
            end_index = _fullstring[start_index:].find("\nEnd\n")
            if end_index == -1:
                run_string += _fullstring[start_index:]
                _fullstring = ""
            else:                
                run_string += _fullstring[start_index:start_index+end_index]
                _fullstring = _fullstring[start_index+end_index:]

        # split strings into time blocks
        log_str_split_by_time = re.split(r"\nTime = (\d+)\n", run_string)

        # get log strings by time
        Times = []
        Time = False
        log_lines_by_time = {}
        for line in log_str_split_by_time:
            regsearch = re.search(r"^\d+", line)
            if regsearch:
                Time = int(regsearch.group())
                Times.append(Time)
            else:
                if Time:
                    log_lines_by_time[Time] = line

        # init residual dict
        residuals = {}
        for parameter in parameters:
            residuals[parameter] = {
                "Initial residual" : [],
                "Final residual" : []
            }
        residuals["Time"] = Times
        residuals["ExecutionTime"] = []
        residuals["ClockTime"] = []
        residuals["bounding nuTilda"] = {
                "min" : [],
                "max" : [],
                "average" : []
            }


        # extract residuals
        for Time in Times:
            log = log_lines_by_time[Time]
            lines = log.split("\n")
            for line in lines:
                #regsearch = re.search(r"[a-zA-Z]:+", log)
                for parameter in parameters:
                    if re.search( f"Solving for {parameter}", line):# Searching for Ux line in file
                        number_string = re.findall(r"Initial residual = (.+?),", line)[0]
                        number = float(number_string)
                        residuals[parameter]["Initial residual"].append(number)
                        number_string = re.findall(r"Final residual = (.+?),", line)[0]
                        number = float(number_string)
                        residuals[parameter]["Final residual"].append(number)
                    elif re.search( f"ExecutionTime = (.+?) s", line):# Searching for Ux line in file
                        number_string = re.findall(f"ExecutionTime = (.+?) s", line)[0]
                        number = float(number_string)
                        residuals["ExecutionTime"].append(number)
                        number_string = re.findall(f"ClockTime = (.+?) s", line)[0]
                        number = float(number_string)
                        residuals["ClockTime"].append(number)

                if re.search( f"bounding nuTilda", line):# bounding nuTilda values line in file
                    number_string = re.findall(f"min: (.+?) ", line)[0]
                    number = float(number_string)
                    residuals["bounding nuTilda"]["min"].append(number)
                    number_string = re.findall(f"max: (.+?) ", line)[0]
                    number = float(number_string)
                    residuals["bounding nuTilda"]["max"].append(number)
                    number_string = re.findall(f"average: (.+?)$", line)[0]
                    number = float(number_string)
                    residuals["bounding nuTilda"]["average"].append(number)
            

        residual_types = ["Final residual", "Initial residual"]
        if len(residuals["ExecutionTime"]) > 0:
            total_execution_time = residuals["ExecutionTime"][-1]
        else:
            total_execution_time = 0.0

        for res in residual_types:
            fig = plt.figure(figsize=(6,6))
            for parameter in parameters:
                line = residuals[parameter][res]
                label = f"{parameter}"
                plt.plot(range(len(line)), line, label=label) # THIS DOES NOT SHOW THE TIMES
                # plt.plot(residuals["Time"], line, label=label)
            plt.title(f"{res} - execution time : {total_execution_time:.2f} s")
            plt.yscale("log")
            plt.legend()
            plt.savefig(os.path.join(self.path_to_postprocessing_folder, f"{res}.png"), dpi=300)
            plt.close()

        parameter = "bounding nuTilda"
        sub_parameters = residuals[parameter].keys()

        fig, axes = plt.subplots(ncols=1, nrows=len(sub_parameters), figsize=(6,6))
        for ax, sub_parameter in zip(axes, sub_parameters):
            line = residuals[parameter][sub_parameter]
            label = f"{sub_parameter}"
            ax.plot(range(len(line)), line) # THIS DOES NOT SHOW THE TIMES
            ax.set_ylabel(label)
            # plt.plot(residuals["Time"], line, label=label)
            ax.set_yscale("symlog")
            ax.grid("both")
        axes[-1].set_xlabel("iteration")
        plt.suptitle(f"{parameter}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_to_postprocessing_folder, f"{parameter}.png"), dpi=300, )
        plt.close()

    def init_target_mean_field(self, target_mean_field : np.array, do_plot : bool = True) -> None:
        """
        """
        self.target_mean_field = target_mean_field
        self.logger.info("imported target mean field")

        if do_plot:
            plot_scalar_field_on_mesh(self.BlockMesh_coordinates_2D[:,0], self.BlockMesh_coordinates_2D[:,1],
                self.target_mean_field[:,0],
                f"target mean field Ux",
                "x", "y", self.get_unit("Ux"), f"target_mean_field_Ux",
                self.path_to_postprocessing_folder,
                close_up_limits=None, cmap="jet", center_cbar=False, aspect_ratio=1)
            plot_scalar_field_on_mesh(self.BlockMesh_coordinates_2D[:,0], self.BlockMesh_coordinates_2D[:,1],
                self.target_mean_field[:,1],
                f"target mean field Uy",
                "x", "y", self.get_unit("Uy"), f"target_mean_field_Uy",
                self.path_to_postprocessing_folder,
                close_up_limits=None, cmap="jet", center_cbar=False, aspect_ratio=1)


    def _filter_by_coordinates(self, unfiltered_data, unfiltered_coordinates, coordinate_filter): 
        """
        filters data by coordinate limits and returns filtered data and filtered coordinates
        """

        filter_indexes = np.where(
            (unfiltered_coordinates[:,0] > coordinate_filter[0][0]) & \
            (unfiltered_coordinates[:,0] < coordinate_filter[0][1]) & \
            (unfiltered_coordinates[:,1] > coordinate_filter[1][0]) & \
            (unfiltered_coordinates[:,1] < coordinate_filter[1][1]))

        filtered_data = unfiltered_data[filter_indexes]
        filtered_data_coordinates = unfiltered_coordinates[filter_indexes]

        return filtered_data, filtered_data_coordinates, filter_indexes

    def set_nutSource(self, time_folder, data, do_plot=True):
        """
        writes nutSource file to specified time folder and writes the data into the folder
        """
        self.logger.info(f"writing nutSource file to time filter {time_folder}")

        if time_folder == "latest":
            folder_content = os.listdir(self.path_to_openfoam_case_folder)
            time_folder = max([int(folder) for folder in folder_content if folder.isnumeric()])

        counter_outlet_nonuniform = 100  # this is hardcoded at the moment! was 150 for old case TODO: dynamically import this counter from mesh (ccy file look for outlet part)

        filename =  "nutSource"
        target_filename = "nutSource"
        path_to_nutSource_template = os.path.join(self.path_to_openfoam_templates, filename)

        destination = os.path.join(self.path_to_openfoam_case_folder, str(time_folder), target_filename)

        with open(path_to_nutSource_template) as f:
            complete_text = f.read()

            complete_text = complete_text.replace("#time_folder", str(time_folder))

            complete_text = complete_text.replace("#counter_internal_field", str(int(self.BlockMesh_coordinates.shape[0])))

            values_internal_field = ""
            for i in range(data.shape[0]):
                values_internal_field += f"{data[i][0]:.6f}\n"

            complete_text = complete_text.replace("#values_internal_field", values_internal_field)

            complete_text = complete_text.replace("#value_inlet_freestreamValue", str(0.0))

            complete_text = complete_text.replace("#value_inlet_value", str(0.0))

            complete_text = complete_text.replace("#value_outlet_freestreamValue", str(0.0))

            complete_text = complete_text.replace("#counter_outlet_nonuniform", str(counter_outlet_nonuniform))

            values_outlet_nonuniform = ""
            for i in range(counter_outlet_nonuniform):
                values_outlet_nonuniform += "0.0\n"

            complete_text = complete_text.replace("#values_outlet_nonuniform", values_outlet_nonuniform)

            complete_text = complete_text.replace("#value_top_value", str(0.0))

        with open(destination, 'w') as f:
            f.write(complete_text)
        
        self.logger.info(f"saved {target_filename} to {destination}")

        if do_plot:
            plot_scalar_field_on_mesh(self.BlockMesh_coordinates_2D[:,0], self.BlockMesh_coordinates_2D[:,1],
                            data,
                            f"nutSource at t = {time_folder}",
                            "x", "y", self.get_unit("nutSource"), f"nutSource_Setting_at_t_{time_folder}",
                            self.path_to_postprocessing_folder,
                            close_up_limits=None, cmap="jet", center_cbar=False, aspect_ratio=1)


    def set_kSource(self, time_folder, data, do_plot=True):
        """
        writes kSource file to specified time folder and writes the data into the folder
        """
        self.logger.info(f"writing kSource file to time filter {time_folder}")

        if time_folder == "latest":
            folder_content = os.listdir(self.path_to_openfoam_case_folder)
            time_folder = max([int(folder) for folder in folder_content if folder.isnumeric()])

        counter_outlet_nonuniform = 100  # this is hardcoded at the moment! was 150 for old case TODO: dynamically import this counter from mesh (ccy file look for outlet part)

        filename =  "kSource"
        target_filename = "kSource"
        path_to_kSource_template = os.path.join(self.path_to_openfoam_templates, filename)

        destination = os.path.join(self.path_to_openfoam_case_folder, str(time_folder), target_filename)

        with open(path_to_kSource_template) as f:
            complete_text = f.read()

            complete_text = complete_text.replace("#time_folder", str(time_folder))

            complete_text = complete_text.replace("#counter_internal_field", str(int(self.BlockMesh_coordinates.shape[0])))

            values_internal_field = ""
            for i in range(data.shape[0]):
                values_internal_field += f"{data[i][0]:.6f}\n"

            complete_text = complete_text.replace("#values_internal_field", values_internal_field)
            complete_text = complete_text.replace("#value_inlet_freestreamValue", str(0.0))
            complete_text = complete_text.replace("#value_inlet_value", str(0.0))
            complete_text = complete_text.replace("#value_outlet_freestreamValue", str(0.0))
            complete_text = complete_text.replace("#counter_outlet_nonuniform", str(counter_outlet_nonuniform))

            values_outlet_nonuniform = ""
            for i in range(counter_outlet_nonuniform):
                values_outlet_nonuniform += "0.0\n"

            complete_text = complete_text.replace("#values_outlet_nonuniform", values_outlet_nonuniform)
            complete_text = complete_text.replace("#value_top_value", str(0.0))

        with open(destination, 'w') as f:
            f.write(complete_text)
        
        self.logger.info(f"saved {target_filename} to {destination}")

        if do_plot:
            plot_scalar_field_on_mesh(self.BlockMesh_coordinates_2D[:,0], self.BlockMesh_coordinates_2D[:,1],
                            data,
                            f"kSource at t = {time_folder}",
                            "x", "y", self.get_unit("kSource"), f"kSource_Setting_at_t_{time_folder}",
                            self.path_to_postprocessing_folder,
                            close_up_limits=None, cmap="jet", center_cbar=False, aspect_ratio=1)



    def calculate_error(self) -> None:
        """
        """

        for time in self.time_list_for_current_import:
            self.logger.info(f"calculating error for t={time}")
            #self.results_by_time[time]["U"]
            simulation_results, simulation_results_coordinates, filter_indexes = self._filter_by_coordinates(
                unfiltered_data = self.results_by_time[time]["U"],
                unfiltered_coordinates = self.BlockMesh_coordinates_2D,
                coordinate_filter = self.error_coordinate_limits)

            target_values, target_value_coordinates, filter_indexes = self._filter_by_coordinates(
                unfiltered_data = self.target_mean_field[:,0:2],
                unfiltered_coordinates = self.BlockMesh_coordinates_2D,
                coordinate_filter = self.error_coordinate_limits)

            self.coordinate_filter_indexes = filter_indexes

            # absolute error: shape: (n_cells, 2)
            absolute_error_by_dir = simulation_results[:,0:2] - target_values[:,0:2] 

            # error_squared_per_cell_per_dir: simply square the error in each direction.  shape: (n_cells, 2)
            error_squared_per_cell_per_dir = absolute_error_by_dir ** 2
            # rmse_per_cell: add the two squared errors per cell and take the sqrt of the sum to get an root squared error per cell
            #  shape: (n_cells, 2)
            rse_per_cell = np.sqrt(np.sum(error_squared_per_cell_per_dir, axis=1))
            # now take the mean over all cells
            rmse = rse_per_cell.mean()
            #print(rmse)

            # area normalized error
            # old
            self.normalized_cell_areas_for_error_calculation = self.cell_areas[filter_indexes] / np.mean(self.cell_areas[filter_indexes])
            #area_weighted_rse_per_cell = rse_per_cell * self.normalized_cell_areas_for_error_calculation 
            #area_weighted_rmse = np.mean(area_weighted_rse_per_cell)

            # new
            abs_error_by_dir = np.abs(absolute_error_by_dir)
            coords_for_error_calc = simulation_results_coordinates
            x_coords_for_error_calc = np.unique(simulation_results_coordinates[:,0])
            areas_x, areas_r = [], []
            area_weighted_rse_per_cell = np.zeros_like(rse_per_cell)
            for x_coord in x_coords_for_error_calc:
                indexes_for_current_x_coord = simulation_results_coordinates[:,0] == x_coord
                y_coords_for_current_x_coord = simulation_results_coordinates[indexes_for_current_x_coord, 1]
                abs_error_for_current_x_coord = abs_error_by_dir[indexes_for_current_x_coord]
                area_x = np.trapz(abs_error_for_current_x_coord[:,0],y_coords_for_current_x_coord)
                area_r = np.trapz(abs_error_for_current_x_coord[:,1],y_coords_for_current_x_coord)
                area_weighted_rse_per_cell[indexes_for_current_x_coord] = area_x + area_r

                areas_x.append(area_x)
                areas_r.append(area_r)

            line_based_error_areas = np.array([areas_x, areas_r]).T
            area_weighted_rmse = np.mean(np.sum(line_based_error_areas, axis=1))
            #####################



            # error = np.sqrt(np.sum(np.abs(absolute_error_by_dir), axis=1)**2)
            # error_old = np.sqrt(np.sum(absolute_error_by_dir, axis=1)**2)
            # rmse = np.sum(error)/error.shape[0]
            # print(rmse)
            # rmse_old = np.sum(error_old)/error_old.shape[0]
            # print(rmse_old)

            # store results
            self.rmse_by_time.append(rmse)
            self.absolute_error_by_dir_by_time.append(absolute_error_by_dir)
            self.error_by_time.append(rse_per_cell)

            self.area_weighted_rmse_by_time.append(area_weighted_rmse)
            self.area_weighted_error_by_time.append(area_weighted_rse_per_cell)

            self.simulation_results_coordinates = simulation_results_coordinates

        self.rmse = self.rmse_by_time[-1]
        self.aw_rmse = self.area_weighted_rmse_by_time[-1]
        
        # calculate min rmse
        if self.rmse < self.min_rmse:
            self.min_rmse = self.rmse
            self.time_of_min_rmse = self.time_list_for_current_import[-1]

    def plot_error(self, enforce_plotting : bool = False ) -> None:
        """
        """

        if (self.config["make_plots"] or enforce_plotting) and not self.last_run_failed:

            # close_up_limits = {
            #     "x" : (float(self.config["close_up_limits_x_lower"]), float(self.config["close_up_limits_x_upper"])),
            #     "y" : (float(self.config["close_up_limits_y_lower"]), float(self.config["close_up_limits_y_upper"]))
            # }

            
            fig = plt.figure()
            plt.plot(self.time_list, self.rmse_by_time/self.rmse_by_time[0])
            plt.xlabel("epoch")
            plt.ylabel("relative rmse")
            if np.max(self.rmse_by_time/self.rmse_by_time[0]) >= 2:
                plt.ylim((0.3, 2))
            plt.grid()
            plt.title(f"relative rmse ({self.config['name_of_run']})\n(rmse at start: {self.rmse_by_time[0]:.4f}, min: {self.min_rmse:.4f} ({100*self.min_rmse/self.rmse_by_time[0]:.1f}%) at t={self.time_of_min_rmse})")
            fig.savefig(self.path_to_postprocessing_folder+"/rmse_by_time.png", dpi=200)

            # save rmse data
            try:    
                df = pd.DataFrame({'epoch': self.time_list, 'relative_rmse': self.rmse_by_time/self.rmse_by_time[0]} ) 
                df.to_csv(self.path_to_postprocessing_folder+"/rmse_by_time.csv")
            except:
                self.logger.warning(f"saving rmse_by_time.csvfailed")


            fig = plt.figure()
            plt.plot(self.time_list, self.area_weighted_rmse_by_time/self.area_weighted_rmse_by_time[0])
            plt.xlabel("epoch")
            plt.ylabel("relative rmse")
            if np.max(self.rmse_by_time/self.rmse_by_time[0]) >= 2:
                plt.ylim((0.3, 2))
            plt.grid()
            plt.title(f"relative area weighted rmse\nat start: {self.area_weighted_rmse_by_time[0]:.4f}, min: {np.min(self.area_weighted_rmse_by_time):.4f} ({100*np.min(self.area_weighted_rmse_by_time)/self.area_weighted_rmse_by_time[0]:.1f}%)")
            fig.savefig(self.path_to_postprocessing_folder+"/rmse_aw_by_time.png", dpi=200)

            fig = plt.figure()
            _t = self.time_list_for_current_import 
            _values = self.area_weighted_rmse_by_time[-len(_t):]/self.area_weighted_rmse_by_time[0]
            plt.plot(_t,_values)
            plt.xlabel("epoch")
            plt.ylabel("relative area weighted rmse")
            plt.grid()
            plt.title(f"relative area weighted rmse\nat start: {self.area_weighted_rmse_by_time[0]:.4f}, min: {np.min(self.area_weighted_rmse_by_time):.4f} ({100*np.min(self.area_weighted_rmse_by_time)/self.area_weighted_rmse_by_time[0]:.1f}%)")
            fig.savefig(self.path_to_postprocessing_folder+f"/rmse_aw_by_time_last_episode_t{self.time_list[-1]}.png", dpi=200)

            error = self.error_by_time[-1]
            rmse = self.rmse_by_time[-1]
            self.logger.info(f"mean squared error: {rmse:.3f}")
            plot_scalar_field_on_mesh(self.simulation_results_coordinates[:,0], self.simulation_results_coordinates[:,1], 
                                    error,f"root squared error per cell (rmse for entire domain = {rmse:.3f})",
                                    "x", "y", "rse", f"error_t{self.time_list[-1]}",
                                    self.path_to_postprocessing_folder,
                                    close_up_limits=None, aspect_ratio=2)
            
            aw_error = self.area_weighted_error_by_time[-1]
            aw_rmse = self.area_weighted_rmse_by_time[-1]
            self.logger.info(f"area weighted mean squared error: {rmse:.3f}")
            plot_scalar_field_on_mesh(self.simulation_results_coordinates[:,0], self.simulation_results_coordinates[:,1], 
                                    aw_error,f"area weighted root squared error per cell (aw_rmse for entire domain = {aw_rmse:.3f})",
                                    "x", "y", "rse", f"aw_error_t{self.time_list[-1]}",
                                    self.path_to_postprocessing_folder,
                                    close_up_limits=None, aspect_ratio=2)
            
            plot_scalar_field_on_mesh(self.simulation_results_coordinates[:,0], self.simulation_results_coordinates[:,1], 
                        self.normalized_cell_areas_for_error_calculation ,f"normalized cell areas",
                        "x", "y", "1", f"normalized_cell_area_for_error_t{self.time_list[-1]}",
                        self.path_to_postprocessing_folder,
                        close_up_limits=None, aspect_ratio=2)

            for i, error_parameter in zip(range(2), ["Ux", "Uy"]):
                absolute_error = self.absolute_error_by_dir_by_time[-1][:,i]

                plot_scalar_field_on_mesh(self.simulation_results_coordinates[:,0], self.simulation_results_coordinates[:,1],
                                            absolute_error,
                                            f"absolute error based on {error_parameter} (rmse = {rmse:.3f})\npositive values mean SA predicts higher values than LES mean field",
                                            "x", "y", "m/s", f"absolute_error_{error_parameter}_t{self.time_list[-1]}",
                                            self.path_to_postprocessing_folder,
                                            close_up_limits=None, cmap="seismic", center_cbar=True, aspect_ratio=2)


        else:
            self.logger.info("no error plots exported because plotting is deactivated")

    
    def plot_results(self, enforce_plotting : bool = False ) -> None:
        """
        """

        if (self.config["make_plots"] or enforce_plotting) and not self.last_run_failed:

            close_up_limits = {
                "x" : (float(self.config["close_up_limits_x_lower"]), float(self.config["close_up_limits_x_upper"])),
                "y" : (float(self.config["close_up_limits_y_lower"]), float(self.config["close_up_limits_y_upper"]))
            }

            for i, param in zip(range(2), ["Ux", "Uy"]):
                plot_scalar_field_on_mesh(self.BlockMesh_coordinates_2D[:,0], self.BlockMesh_coordinates_2D[:,1],
                                            self.results["U"][:,i],
                                            f"simulation results: {param}",
                                            "x", "y", self.get_unit(param), f"simulation_results_{param}_t{self.time_list[-1]}",
                                            self.path_to_postprocessing_folder,
                                            close_up_limits=None, close_up_aspect_ratio=2, cmap="jet", center_cbar=False, aspect_ratio=None)
            
            params = list(self.results.keys())
            params.remove("U") # to plot the other parameters remove U because it has a different shape
            for param in params:  
                plot_scalar_field_on_mesh(self.BlockMesh_coordinates_2D[:,0], self.BlockMesh_coordinates_2D[:,1],
                                            self.results[param],
                                            f"simulation results: {param}",
                                            "x", "y", self.get_unit(param),  f"simulation_results_{param}_t{self.time_list[-1]}",
                                            self.path_to_postprocessing_folder,
                                            close_up_limits=None, close_up_aspect_ratio=2, cmap="jet", center_cbar=False, aspect_ratio=None)

        else:
            self.logger.info("no error plots exported because plotting is deactivated")


    def split_cells_into_batches_rectanglefit(self) -> tuple: 
        """
        """
        # cluster variant

        self.logger.info("now splitting cells into batches using the automatic rectangle fit")
        self.n_cells = self.BlockMesh_coordinates_2D.shape[0]

        # determine optimal rectangle size
        unique_x = np.sort(np.unique(self.BlockMesh_coordinates_2D[:,0]))
        self.logger.info(f"the mesh contains {unique_x.shape[0]} cells in x-direction")
        # find the number of cells in y direction for each unique x coordinate
        unique_y_per_unique_x = []
        for x_coor in unique_x:
            inds_found = np.where((self.BlockMesh_coordinates_2D[:,0] == x_coor))
            unique_y_per_unique_x.append( np.unique(self.BlockMesh_coordinates_2D[inds_found,1]).shape[0])
        # compute greatest common divisor of the number of cells in x direction and the number of cells in y direction
        relevant_cell_counts = list(set(unique_y_per_unique_x))+ [unique_x.shape[0]]
        self.logger.info(f"the mesh contains {list(set(unique_y_per_unique_x))} cells in y-direction")
        cells_per_batch_block_side = np.gcd.reduce(relevant_cell_counts)
        self.logger.info(f"the optimal rectangle side length ist threfore {cells_per_batch_block_side} cells")
        cells_per_batch = cells_per_batch_block_side * cells_per_batch_block_side
        self.logger.info(f"each batch will therefore contain {cells_per_batch} cells")


        # copy the original list of coorinates
        COORD_COPY = self.BlockMesh_coordinates_2D.copy()

        # init batch lists
        self.batch_x_coordinates = []  # list of arrays
        self.batch_y_coordinates = []  # list of arrays
        self.batch_index_list = []

        # split array
        list_of_x_coords = [ unique_x[i*cells_per_batch_block_side : (i+1)*cells_per_batch_block_side] for i in range(int(unique_x.shape[0]/cells_per_batch_block_side))]

        batch_counter = 0

        for x_batch_counter, x_coords in enumerate(list_of_x_coords):
            #print(f"x coordinate batch number: {x_batch_counter}")

            # first, cout how many cell remain for the current selection of x coordinates
            remaining_cells_for_current_x_coordinates = 0
            for x_coor in x_coords:
                inds_found = np.where((COORD_COPY[:,0] == x_coor))
                remaining_cells_for_current_x_coordinates += len(inds_found[0])

            # construct batches only as long there are enough cells remaining
            while remaining_cells_for_current_x_coordinates >= cells_per_batch:
                _this_batch_x_coords = []
                _this_batch_y_coords = []
                # loopg thourgh each single x coordinate
                for x_coor in x_coords:
                    # find remaining y coordinates at this x position 
                    ind = np.where((COORD_COPY[:,0] == x_coor))[0]  # get indexes
                    y_coords = COORD_COPY[ind,1]  # get y coords
                    sorting_inds = np.argsort(y_coords) # get the indexes that sort the y coordinates to enforce a consistent intra batch order
                    y_coords_for_selection = y_coords[sorting_inds][0:cells_per_batch_block_side]
                    indexes_to_pop = ind[sorting_inds][0:cells_per_batch_block_side]
                    COORD_COPY = np.delete(COORD_COPY , indexes_to_pop, axis=0)  # remove the coordinates that will now be added to a new batch
                    _this_batch_x_coords.extend(np.array(cells_per_batch_block_side * [x_coor]))
                    _this_batch_y_coords.extend(y_coords_for_selection)
                
                if (len(_this_batch_x_coords) == cells_per_batch) and (len(_this_batch_y_coords) == cells_per_batch):
                    self.batch_x_coordinates.append(np.array(_this_batch_x_coords).reshape((cells_per_batch,1)))
                    self.batch_y_coordinates.append(np.array(_this_batch_y_coords).reshape((cells_per_batch,1)))
                    
                    # find indexes of current batch entries in original coordinate array
                    index_list_for_current_batch = []
                    for _x, _y in zip(_this_batch_x_coords, _this_batch_y_coords):
                        x_ind = np.where(self.BlockMesh_coordinates_2D[:,0] == _x)[0]
                        y_ind = np.where(self.BlockMesh_coordinates_2D[:,1] == _y)[0]
                        np.intersect1d(x_ind, y_ind)[0]
                        index_list_for_current_batch += [np.intersect1d(x_ind, y_ind)[0]]
                    self.batch_index_list.append(index_list_for_current_batch)

                batch_counter += 1
                #print(f"created batch number: {batch_counter}")

                # count how many cell remain for the current selection of x coordinates
                remaining_cells_for_current_x_coordinates = 0
                for x_coor in x_coords:
                    inds_found = np.where((COORD_COPY[:,0] == x_coor))
                    remaining_cells_for_current_x_coordinates += len(inds_found[0])
        
        self.logger.info(f"constructed {batch_counter} cell batches")

        fig = plt.figure()
        for batch_id, (_x_list, _y_list) in enumerate(zip(self.batch_x_coordinates,  self.batch_y_coordinates)):
            plt.scatter(_x_list[:,0], _y_list[:,0], label=str(batch_id), s=1)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"batch coordinates")
        plt.savefig(f"{self.path_to_postprocessing_folder}/all_batches_scatterplot.png", dpi=300)
        plt.ylim([0, 0.05])
        plt.xlim([-0.12, 0.1])
        plt.savefig(f"{self.path_to_postprocessing_folder}/all_batches_scatterplot_closeup.png", dpi=300)
        plt.close()

        for batch_id in np.linspace(start=0, stop=batch_counter-1, num=5, dtype=int):
            _x_list = self.batch_x_coordinates[batch_id][:]
            _y_list = self.batch_y_coordinates[batch_id][:]
            fig = plt.figure()
            for cell_id, (_x, _y) in enumerate(zip(_x_list, _y_list)):
                plt.scatter(_x, _y, label=str(cell_id), s=20)
                plt.text(_x, _y, s=str(cell_id))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"batch array indexes of batch {batch_id}")
            plt.savefig(f"{self.path_to_postprocessing_folder}/batch_{batch_id}.png", dpi=300)
            plt.close()


        # extract batch id per cell
        self.batch_id_per_cell = -1*np.ones(self.BlockMesh_coordinates_2D[:,0].shape)

        for batch_id, (_x_list, _y_list) in enumerate(zip(self.batch_x_coordinates,  self.batch_y_coordinates)):
            #print(f"batch {batch_id}")
            for _x, _y in zip(_x_list[:,0], _y_list[:,0]):
                x_inds = np.argwhere(self.BlockMesh_coordinates_2D[:,0]==_x)
                y_inds = np.argwhere(self.BlockMesh_coordinates_2D[:,1]==_y)
                matches = np.intersect1d(x_inds, y_inds)
                if (matches.shape[0]==1):
                    self.batch_id_per_cell[matches[0]] = batch_id + 1
                else:
                    print("error")
        
        plot_scalar_field_on_mesh(self.BlockMesh_coordinates_2D[:,0], self.BlockMesh_coordinates_2D[:,1],
            self.batch_id_per_cell,
            f"mesh cell batches - batches",
            "x", "y", "batch id", f"cell_batches",
            self.path_to_postprocessing_folder,
            close_up_limits=None, cmap="flag", center_cbar=False, aspect_ratio=1,
            custom_axis_limits=None, levels_overwrite=batch_counter)
        
        self.logger.info(f"finished constructed {batch_counter} cell batches")

        return self.batch_index_list, self.batch_x_coordinates, self.batch_y_coordinates


    def init_neighborhood_batches_OLD_VERSION(self, rectangle_length : float, points_per_batch_side : int):
        """
        This is the new method of domain splitting. Every cell in the domain will be the center of a interpolation batch.
        """

        self.logger.info("now starting neighborhood domain splitting")

        self.rectangle_length = rectangle_length
        self.points_per_batch_side = points_per_batch_side
        pad = 0.5*self.rectangle_length
        padding_wide = pad # 30*self.rectangle_length
        wide_padding_applied_counter = 0
        _status_logging_frequency = 1000

        # the error is only defined on certain coordinate, to include the error as a value for interpolation it has to be enlarged to the full grid size
        last_error_projected_onto_full_grid = np.zeros(shape=(self.results["U"].shape[0],1))
        last_error_projected_onto_full_grid[self.coordinate_filter_indexes] = self.error_by_time[-1].reshape(self.error_by_time[-1].shape[0], 1)

        values = np.hstack((self.results["U"][:,0:2], last_error_projected_onto_full_grid, self.results["nut"]))

        # apply boundary conditions
        additional_bc_cells, additional_bc_values = self.apply_boundary_conditions(values)

        self.cell_ids = np.arange(self.BlockMesh_coordinates_2D.shape[0])
        self.cell_batch_data = {}
        self.cell_batches = []

        _cellbatch_loop_start = time.time()
        _timer = time.time()
        bc_interpolation_counter = 0
        # loop through every cell
        for cell_id in self.cell_ids:

            wide_padding_applied = False
            padding_wide = pad
            
            if cell_id % _status_logging_frequency == 0:
                _duration = time.time() - _timer
                _timer = time.time()
                _time_per_cell = _duration/_status_logging_frequency
                self.logger.info(f"initializing cell batches for cell {cell_id}/{len(self.cell_ids)}, average time per cell batch: {1000*_time_per_cell:.2f} s/1000 cell batches, time left: {_time_per_cell*(len(self.cell_ids)-cell_id)/60:.2f} minutes, wide padding applied to: {wide_padding_applied_counter} cells")

            #_timer = time.time()
            self.cell_batch_data[cell_id] = {}
            center_coordinates = self.BlockMesh_coordinates_2D[cell_id]
            x_batch_range = [center_coordinates[0]-self.rectangle_length/2, center_coordinates[0]+self.rectangle_length/2]
            y_batch_range = [center_coordinates[1]-self.rectangle_length/2, center_coordinates[1]+self.rectangle_length/2]
            #print(f"timer:  {self.cell_ids.shape[0]*(time.time()-_timer):.3f} seconds")

            # create the 'real' meshgrid for the cell and its neighborhood
            #_timer = time.time()
            x_batch_real = np.linspace(x_batch_range[0], x_batch_range[1], num=self.points_per_batch_side)
            y_batch_real = np.linspace(y_batch_range[0], y_batch_range[1], num=self.points_per_batch_side)
            meshgrid_xx, meshgrid_yy = np.meshgrid(x_batch_real, y_batch_real)
            #print(f"timer:  {self.cell_ids.shape[0]*(time.time()-_timer):.3f} seconds")

            # get cells that are in the wider region (plus padding) 
            #_timer = time.time()
            point_inside_x_wider_region = np.logical_and((self.BlockMesh_coordinates_2D[:,0] >= np.min(x_batch_real)-pad), (self.BlockMesh_coordinates_2D[:,0] <= np.max(x_batch_real)+pad))
            point_inside_y_wider_region = np.logical_and((self.BlockMesh_coordinates_2D[:,1] >= np.min(y_batch_real)-pad), (self.BlockMesh_coordinates_2D[:,1] <= np.max(y_batch_real)+pad))
            cells_in_wider_region = np.nonzero(np.logical_and(point_inside_x_wider_region, point_inside_y_wider_region))[0]  # the nonzero part reduces the shape from 77800 to e.g. 11000
            
            while cells_in_wider_region.shape[0] < (self.points_per_batch_side**2)*1.6:
                # try again with larger padding
                padding_wide = padding_wide*2
                point_inside_x_wider_region = np.logical_and((self.BlockMesh_coordinates_2D[:,0] >= np.min(x_batch_real)-padding_wide), (self.BlockMesh_coordinates_2D[:,0] <= np.max(x_batch_real)+padding_wide))
                point_inside_y_wider_region = np.logical_and((self.BlockMesh_coordinates_2D[:,1] >= np.min(y_batch_real)-padding_wide), (self.BlockMesh_coordinates_2D[:,1] <= np.max(y_batch_real)+padding_wide))
                cells_in_wider_region = np.nonzero(np.logical_and(point_inside_x_wider_region, point_inside_y_wider_region))[0]  # the nonzero part reduces the shape from 77800 to e.g. 11000
                #print(cells_in_wider_region.shape[0])
                wide_padding_applied = True

            if wide_padding_applied:
                wide_padding_applied_counter += 1
            #print(f"timer:  {self.cell_ids.shape[0]*(time.time()-_timer):.3f} seconds")

            # get the griddata based on the values (U velocity)
            #_timer = time.time()
            cell_coordinates_wider_region = self.BlockMesh_coordinates_2D[cells_in_wider_region]
            data_of_wider_region = values[cells_in_wider_region]
            #print(f"timer:  {self.cell_ids.shape[0]*(time.time()-_timer):.3f} seconds")

            # first try interpolation with selected data (no boundary contitions)
            try:
                #_timer = time.time()
                cell_batch_interpolated_griddata = griddata(cell_coordinates_wider_region, data_of_wider_region, (meshgrid_xx, meshgrid_yy), method=self.interpolation_method_for_cell_batches)  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
                #print(f"timer:  {self.cell_ids.shape[0]*(time.time()-_timer)/60:.3f} min")

                if np.isnan(np.sum(cell_batch_interpolated_griddata)):
                    # np.nan is in interpolation data, this probably means it is a cell at the boundary
                    # add boundary points
                    cell_coordinates_wider_region = np.vstack((cell_coordinates_wider_region, additional_bc_cells))
                    data_of_wider_region = np.vstack((data_of_wider_region, additional_bc_values))
                    cell_batch_interpolated_griddata = griddata(cell_coordinates_wider_region, data_of_wider_region, (meshgrid_xx, meshgrid_yy), method=self.interpolation_method_for_cell_batches)  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
                    bc_interpolation_counter += 1
                    if np.isnan(np.sum(cell_batch_interpolated_griddata)):
                        # although boundary conditions where added, there is still np.nan in the interpolation, plot data and use nearest interpolation
                        plot_interpolation_as_scatterplot(cell_batch_interpolated_griddata , x_batch_real, y_batch_real, cell_coordinates_wider_region, data_of_wider_region, 
                            "the interpolation contained np.nan although bc were applied",
                            f"cell_batch_{cell_id}_failed.png",
                            self.path_to_postprocessing_folder, self.logger)
                        
                        cell_batch_interpolated_griddata = griddata(cell_coordinates_wider_region, data_of_wider_region, (meshgrid_xx, meshgrid_yy), method='nearest')  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
                    else:
                        if (bc_interpolation_counter % 100 == 0) and not self.cell_batches_plotted:
                            # plot everey 100th interpolation on boundary 
                            plot_interpolation_as_scatterplot(cell_batch_interpolated_griddata , x_batch_real, y_batch_real, cell_coordinates_wider_region, data_of_wider_region, 
                                "interpolation data, boundary conditions applied",
                                f"cell_batch_{cell_id}_bc.png",
                                self.path_to_postprocessing_folder, self.logger)
            except:
                self.logger.warning(f"fatal error occured during interpolation, using nearest for cell id: {cell_id}")
                # fata error occured during interpolation, use nearest interpolation and plot data 
                cell_batch_interpolated_griddata = griddata(cell_coordinates_wider_region, data_of_wider_region, (meshgrid_xx, meshgrid_yy), method='nearest')  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
                plot_interpolation_as_scatterplot(cell_batch_interpolated_griddata , x_batch_real, y_batch_real, cell_coordinates_wider_region, data_of_wider_region, 
                    "the interpolation failed",
                    f"cell_batch_{cell_id}_fatal_error.png",
                    self.path_to_postprocessing_folder, self.logger)
                

            #######################################################################################################

            # compute interpolated grid
            #timer = time.time()
            #print(f"timer:  {self.cell_ids.shape[0]*(time.time()-_timer):.3f} seconds")

            # if np.isnan(np.sum(cell_batch_interpolated_griddata)):
            #     data_of_wider_region
            #     cell_coordinates_wider_region

            #_timer = time.time()
            self.cell_batches.append((x_batch_real, y_batch_real, cell_batch_interpolated_griddata))
            #print(f"timer:  {self.cell_ids.shape[0]*(time.time()-_timer):.3f} seconds")

            if (cell_id % 20000) == 0 and not self.cell_batches_plotted:
                plot_interpolated_cell_batch(cell_batch_interpolated_griddata , x_batch_real, y_batch_real,
                                self.BlockMesh_coordinates_2D, values, 
                                "comparison of cell batch interpolation to original data",
                                f"cell_batch_{cell_id}.png",
                                self.path_to_postprocessing_folder, self.logger)

            # SAVE DATA
            # center_coordinates  # (2,0)
            # x_batch_range  # list
            # y_batch_range  # list
            # x_batch_real  # (5,0)
            # y_batch_real  # (5,0)
            # meshgrid_xx  # (5,5)
            # meshgrid_yy  # (5,5)
            # cells_in_wider_region  # (16224)
            # cell_coordinates_wider_region
            # cell_batch_interpolated_griddata

        self.cell_batches_plotted = True
        _cellbatch_loop_end = time.time()
        _cellbatch_loop_duration = _cellbatch_loop_end - _cellbatch_loop_start
        self.logger.info(f"TIMER: cell batch creation loop took: {_cellbatch_loop_duration/60:.1f} minutes")



    def get_neighborhood_batches(self):
        """
        This is the new method of domain splitting. Every cell in the domain will be the center of a interpolation batch.
        """
        self.logger.info("now starting neighborhood domain splitting")

        values = np.hstack((self.results["U"][:,0:2], self.results["nut"]))

        # apply boundary conditions
        additional_bc_cells, additional_bc_values = self.apply_boundary_conditions(values)

        #self.cell_ids = np.arange(self.BlockMesh_coordinates_2D.shape[0])
        self.cell_batch_data = {}
        self.cell_batches = []

        _cellbatch_loop_start = time.time()
        _timer = time.time()
        bc_interpolation_counter = 0
        _status_logging_frequency = 100
        # self.cell_batches_plotted = True
        # loop through every cell
        for cell_id in self.cell_ids_selected:

            # first try interpolation with selected data (no boundary contitions)
            try:
                if cell_id % _status_logging_frequency == 0:
                    _duration = time.time() - _timer
                    _timer = time.time()
                    _time_per_cell = _duration/_status_logging_frequency
                    self.logger.info(f"getting cell batches for cell {cell_id}/{len(self.cell_ids_selected)}, average time per cell batch: {1000*_time_per_cell:.2f} s/1000 cell batches, time left: {_time_per_cell*(len(self.cell_ids_selected)-cell_id)/60:.2f} minutes. interpolation counter: {bc_interpolation_counter}")

                data_of_wider_region = values[self.cell_dict.get(cell_id, "cell_IDs_in_wider_region")]
                if self.cell_dict.get(cell_id, "boundary conditions apply"):
                    data_of_wider_region = np.vstack((data_of_wider_region, additional_bc_values))
                interpolator = LinearNDInterpolator(self.cell_dict.get(cell_id, "triangulation"), data_of_wider_region)
                cell_batch_interpolated_griddata = interpolator(self.cell_dict.get(cell_id, "new mesh"))

                if np.isnan(np.sum(cell_batch_interpolated_griddata)):
                    # np.nan is in interpolation data, this probably means it is a cell at the boundary
                    # add boundary points
                    self.cell_dict.set(cell_id, "boundary conditions apply", True)
                    cell_coordinates_wider_region = self.cell_dict.get(cell_id, "cell_coordinates_wider_region")
                    cell_coordinates_wider_region = np.vstack((cell_coordinates_wider_region, additional_bc_cells))
                    data_of_wider_region = np.vstack((data_of_wider_region, additional_bc_values))
                    self.cell_dict.set(cell_id, "cell_coordinates_wider_region", cell_coordinates_wider_region)
                    self.cell_dict.set(cell_id, "triangulation", Delaunay(cell_coordinates_wider_region))
                    #cell_batch_interpolated_griddata = griddata(cell_coordinates_wider_region, data_of_wider_region, (meshgrid_xx, meshgrid_yy), method=self.interpolation_method_for_cell_batches)  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
                    interpolator = LinearNDInterpolator(self.cell_dict.get(cell_id, "triangulation"), data_of_wider_region)
                    cell_batch_interpolated_griddata = interpolator(self.cell_dict.get(cell_id, "new mesh"))

                    bc_interpolation_counter += 1
                    if np.isnan(np.sum(cell_batch_interpolated_griddata)):
                        # although boundary conditions where added, there is still np.nan in the interpolation, plot data and use nearest interpolation
                        plot_interpolation_as_scatterplot(cell_batch_interpolated_griddata , self.cell_dict.get(cell_id, "x_batch_real"),
                            self.cell_dict.get(cell_id, "y_batch_real"),
                            cell_coordinates_wider_region, data_of_wider_region, 
                            "the interpolation contained np.nan although bc were applied",
                            f"cell_batch_{cell_id}_failed.png",
                            self.path_to_postprocessing_folder, self.logger)
                        
                        cell_batch_interpolated_griddata = griddata(cell_coordinates_wider_region, data_of_wider_region, self.cell_dict.get(cell_id, "new mesh"), method='nearest')  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
                    else:
                        if (bc_interpolation_counter % 500 == 0) and not self.cell_batches_plotted:
                            plot_interpolation_new(
                                self.BlockMesh_coordinates_2D,
                                values,
                                cell_batch_interpolated_griddata,
                                self.cell_dict.get(cell_id, "x_batch_real"),
                                self.cell_dict.get(cell_id, "y_batch_real"),
                                cell_coordinates_wider_region,
                                data_of_wider_region, 
                                "stencil",
                                f"cell_batch_{cell_id}_new_plot.png",
                                self.path_to_postprocessing_folder, self.logger)
                            # plot everey 100th interpolation on boundary 
                            plot_interpolation_as_scatterplot(cell_batch_interpolated_griddata , self.cell_dict.get(cell_id, "x_batch_real"),
                                self.cell_dict.get(cell_id, "y_batch_real"), cell_coordinates_wider_region, data_of_wider_region, 
                                "interpolation data, boundary conditions applied",
                                f"cell_batch_{cell_id}_bc.png",
                                self.path_to_postprocessing_folder, self.logger)
                # # TODO: remove - this ist just for plotting
                # else:
                #     # if it worked right away
                #     if cell_id % 100 == 0:
                #         plot_interpolation_new(
                #             self.BlockMesh_coordinates_2D,
                #             values,
                #             cell_batch_interpolated_griddata,
                #             self.cell_dict.get(cell_id, "x_batch_real"),
                #             self.cell_dict.get(cell_id, "y_batch_real"),
                #             cell_coordinates_wider_region,
                #             data_of_wider_region, 
                #             "stencil",
                #             f"cell_batch_{cell_id}_new_plot.png",
                #             self.path_to_postprocessing_folder, self.logger)
            except:
                self.logger.warning(f"fatal error occured during interpolation, using nearest for cell id: {cell_id}")
                # fata error occured during interpolation, use nearest interpolation and plot data 
                cell_batch_interpolated_griddata = griddata(cell_coordinates_wider_region, data_of_wider_region, self.cell_dict.get(cell_id, "new mesh"), method='nearest')  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
                plot_interpolation_as_scatterplot(cell_batch_interpolated_griddata , self.cell_dict.get(cell_id, "x_batch_real"), self.cell_dict.get(cell_id, "y_batch_real"), cell_coordinates_wider_region, data_of_wider_region, 
                    "the interpolation failed",
                    f"cell_batch_{cell_id}_fatal_error.png",
                    self.path_to_postprocessing_folder, self.logger)
                

            #######################################################################################################

            #_timer = time.time()
            self.cell_batches.append((self.cell_dict.get(cell_id, "x_batch_real"), self.cell_dict.get(cell_id, "y_batch_real"), cell_batch_interpolated_griddata))
            #print(f"timer:  {self.cell_ids.shape[0]*(time.time()-_timer):.3f} seconds")

            if (cell_id % 2000) == 0 and not self.cell_batches_plotted:
                plot_interpolated_cell_batch(cell_batch_interpolated_griddata , self.cell_dict.get(cell_id, "x_batch_real"), self.cell_dict.get(cell_id, "y_batch_real"),
                                self.BlockMesh_coordinates_2D, values, 
                                "comparison of cell batch interpolation to original data",
                                f"cell_batch_{cell_id}.png",
                                self.path_to_postprocessing_folder, self.logger)


        self.cell_batches_plotted = True
        _cellbatch_loop_end = time.time()
        _cellbatch_loop_duration = _cellbatch_loop_end - _cellbatch_loop_start
        self.cell_dict.dump_dict()
        self.logger.info(f"TIMER: cell batch creation loop took: {_cellbatch_loop_duration/60:.1f} minutes")

    def get_dummy_batches_for_testing(self, rectangle_length : float, points_per_batch_side : int):
        """
        THIS METHOD IS FOR TESTING AND DEBUGGING PURPOSES ONLY!
        this method creates the cell_batches list that is required by the RL agent. It creates random dummy data that has the correct shape. 
        it is usually created by calling init_neighborhood_batches() and get__neighborhood_batches()
        """ 

        self.cell_ids = np.arange(self.BlockMesh_coordinates_2D.shape[0])
        self.cell_batches = []
        self.rectangle_length = rectangle_length
        self.points_per_batch_side = points_per_batch_side

        self.cell_ids_selected = self.cell_ids[::self.subsampling_ratio]
        self.coordinates_of_selected_cell_ids = [self.BlockMesh_coordinates_2D[cell_id] for cell_id in self.cell_ids_selected]

        # loop through every cell
        for cell_id in self.cell_ids_selected:
            center_coordinates = self.BlockMesh_coordinates_2D[cell_id]
            x_batch_range = [center_coordinates[0]-self.rectangle_length/2, center_coordinates[0]+self.rectangle_length/2]
            y_batch_range = [center_coordinates[1]-self.rectangle_length/2, center_coordinates[1]+self.rectangle_length/2]
            # create the 'real' meshgrid for the cell and its neighborhood
            x_batch_real = np.linspace(x_batch_range[0], x_batch_range[1], num=self.points_per_batch_side)
            y_batch_real = np.linspace(y_batch_range[0], y_batch_range[1], num=self.points_per_batch_side)

            interpolated_griddata_dummy  = np.random.random(size=(points_per_batch_side, points_per_batch_side, 3)) 
            self.cell_batches.append((x_batch_real, y_batch_real, interpolated_griddata_dummy))

        
    def init_neighborhood_batches(self, rectangle_length : float, points_per_batch_side : int):
        """
        This is the new method of domain splitting. Every cell in the domain will be the center of a interpolation batch.
        """

        self.logger.info("now starting neighborhood domain splitting")

        self.rectangle_length = rectangle_length
        self.points_per_batch_side = points_per_batch_side
        pad = 0.5*self.rectangle_length
        padding_wide = pad # 30*self.rectangle_length
        wide_padding_applied_counter = 0
        max_cells_per_batch = 2000
        _status_logging_frequency = 1000

        values = np.hstack((self.results["U"][:,0:2], self.results["nut"]))

        # apply boundary conditions
        # additional_bc_cells, additional_bc_values = self.apply_boundary_conditions(values)

        self.cell_ids = np.arange(self.BlockMesh_coordinates_2D.shape[0])

        self.cell_ids_selected = self.cell_ids[::self.subsampling_ratio]
        self.coordinates_of_selected_cell_ids = [self.BlockMesh_coordinates_2D[cell_id] for cell_id in self.cell_ids_selected]


        #self.cell_batch_data = {}
        #self.cell_batches = []

        _cellbatch_loop_start = time.time()
        _timer = time.time()
        bc_interpolation_counter = 0
        # loop through every cell
        for cell_id in self.cell_ids_selected:

            self.cell_dict.set(cell_id, {})
            
            wide_padding_applied = False
            padding_wide = pad
            
            if cell_id % _status_logging_frequency == 0:
                _duration = time.time() - _timer
                _timer = time.time()
                _time_per_cell = _duration/_status_logging_frequency
                self.logger.info(f"initializing cell batches for cell {cell_id}/{len(self.cell_ids_selected)}, average time per cell batch: {1000*_time_per_cell:.2f} s/1000 cell batches, time left: {_time_per_cell*(len(self.cell_ids_selected)-cell_id)/60:.2f} minutes, wide padding applied to: {wide_padding_applied_counter} cells")

            #_timer = time.time()
            #self.cell_batch_data[cell_id] = {}
            center_coordinates = self.BlockMesh_coordinates_2D[cell_id]
            x_batch_range = [center_coordinates[0]-self.rectangle_length/2, center_coordinates[0]+self.rectangle_length/2]
            y_batch_range = [center_coordinates[1]-self.rectangle_length/2, center_coordinates[1]+self.rectangle_length/2]
            #print(f"timer:  {self.cell_ids.shape[0]*(time.time()-_timer):.3f} seconds")

            # create the 'real' meshgrid for the cell and its neighborhood
            #_timer = time.time()
            x_batch_real = np.linspace(x_batch_range[0], x_batch_range[1], num=self.points_per_batch_side)
            y_batch_real = np.linspace(y_batch_range[0], y_batch_range[1], num=self.points_per_batch_side)
            meshgrid_xx, meshgrid_yy = np.meshgrid(x_batch_real, y_batch_real)
            #print(f"timer:  {self.cell_ids.shape[0]*(time.time()-_timer):.3f} seconds")

            # get cells that are in the wider region (plus padding) 
            #_timer = time.time()
            point_inside_x_wider_region = np.logical_and((self.BlockMesh_coordinates_2D[:,0] >= np.min(x_batch_real)-pad), (self.BlockMesh_coordinates_2D[:,0] <= np.max(x_batch_real)+pad))
            point_inside_y_wider_region = np.logical_and((self.BlockMesh_coordinates_2D[:,1] >= np.min(y_batch_real)-pad), (self.BlockMesh_coordinates_2D[:,1] <= np.max(y_batch_real)+pad))
            cells_in_wider_region = np.nonzero(np.logical_and(point_inside_x_wider_region, point_inside_y_wider_region))[0]  # the nonzero part reduces the shape from 77800 to e.g. 11000
            cell_coordinates_wider_region = self.BlockMesh_coordinates_2D[cells_in_wider_region]

            try:
                triangulation = Delaunay(cell_coordinates_wider_region)  # Compute the triangulation
                triangulation_failed = False
            except QhullError:
                triangulation_failed = True

            while triangulation_failed and cells_in_wider_region.shape[0] < max_cells_per_batch:
                # try again with larger padding
                padding_wide = padding_wide*2
                point_inside_x_wider_region = np.logical_and((self.BlockMesh_coordinates_2D[:,0] >= np.min(x_batch_real)-padding_wide), (self.BlockMesh_coordinates_2D[:,0] <= np.max(x_batch_real)+padding_wide))
                point_inside_y_wider_region = np.logical_and((self.BlockMesh_coordinates_2D[:,1] >= np.min(y_batch_real)-padding_wide), (self.BlockMesh_coordinates_2D[:,1] <= np.max(y_batch_real)+padding_wide))
                cells_in_wider_region = np.nonzero(np.logical_and(point_inside_x_wider_region, point_inside_y_wider_region))[0]  # the nonzero part reduces the shape from 77800 to e.g. 11000
                #print(cells_in_wider_region.shape[0])
                wide_padding_applied = True
                cell_coordinates_wider_region = self.BlockMesh_coordinates_2D[cells_in_wider_region]
                try:
                    triangulation = Delaunay(cell_coordinates_wider_region)  # Compute the triangulation
                    triangulation_failed = False
                except:
                    triangulation_failed = True

            if wide_padding_applied:
                wide_padding_applied_counter += 1
            #print(f"timer:  {self.cell_ids.shape[0]*(time.time()-_timer):.3f} seconds")
            # get the griddata based on the values (U velocity)
            #_timer = time.time()
            #print(f"timer:  {self.cell_ids.shape[0]*(time.time()-_timer):.3f} seconds")
            self.cell_dict.set(cell_id, "cell_coordinates_wider_region", cell_coordinates_wider_region)
            self.cell_dict.set(cell_id, "cell_IDs_in_wider_region", cells_in_wider_region)
            #self.cell_dict[cell_id]["data_of_wider_region"] = data_of_wider_region
            self.cell_dict.set(cell_id, "new mesh", (meshgrid_xx, meshgrid_yy))
            self.cell_dict.set(cell_id, "x_batch_real", x_batch_real)
            self.cell_dict.set(cell_id, "y_batch_real", y_batch_real)
            self.cell_dict.set(cell_id, "boundary conditions apply", False)
            data_of_wider_region = values[cells_in_wider_region]

            # temporay TODO
            # if cell_id % 100 == 0:
            #     cell_batch_interpolated_griddata = griddata(cell_coordinates_wider_region, data_of_wider_region, (meshgrid_xx, meshgrid_yy), method='nearest')  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
            #     plot_interpolation_new(self.BlockMesh_coordinates_2D, values, cell_batch_interpolated_griddata , x_batch_real, y_batch_real, cell_coordinates_wider_region, data_of_wider_region, 
            #         "stencil",
            #         f"cell_batch_{cell_id}_new_plot.png",
            #         self.path_to_postprocessing_folder, self.logger)

            if triangulation_failed:
                self.logger.warning(f"fatal error occured during triangulation, using nearest for cell id: {cell_id}")
                cell_batch_interpolated_griddata = griddata(cell_coordinates_wider_region, data_of_wider_region, (meshgrid_xx, meshgrid_yy), method='nearest')  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
                plot_interpolation_as_scatterplot(cell_batch_interpolated_griddata , x_batch_real, y_batch_real, cell_coordinates_wider_region, data_of_wider_region, 
                    "the interpolation failed",
                    f"cell_batch_{cell_id}_fatal_error.png",
                    self.path_to_postprocessing_folder, self.logger)
            else:
                self.cell_dict.set(cell_id, "triangulation", triangulation)
            
        #self.cell_batches_plotted = True
        _cellbatch_loop_end = time.time()
        _cellbatch_loop_duration = _cellbatch_loop_end - _cellbatch_loop_start
        self.logger.info(f"TIMER: cell batch creation loop took: {_cellbatch_loop_duration/60:.1f} minutes")

    def apply_boundary_conditions(self, values):
        """
        apply boundary conditions
        """
        self.logger.info("now applying boundary conditions")
        _timer = time.time()

        _cell_layers = []
        _cell_values_list = []

        # get lower boundary (axis) - Symmetry Boundary condition
        sorted_unique_y_coords = np.sort(np.unique(self.BlockMesh_coordinates_2D[:,1]))
        half_stencil_length = self.rectangle_length/2
        # the number of cells that will be added below the mesh shall be as thick as half a stencil length
        y_cell_range = 0
        n_new_layers = 0
        while y_cell_range < half_stencil_length:
            n_new_layers += 1
            y_coords_of_lower_cell_rows = sorted_unique_y_coords[0:n_new_layers]
            y_cell_range = np.max(y_coords_of_lower_cell_rows) - np.min(y_coords_of_lower_cell_rows)
        _cell_values = []
        for y_coord in y_coords_of_lower_cell_rows:
            y_coord_of_new_cell_layer = -y_coord
            cell_row_boundary_indices = np.where(self.BlockMesh_coordinates_2D[:,1] == y_coord)[0]
            x_coordinates_of_cell_row_boundary = self.BlockMesh_coordinates_2D[cell_row_boundary_indices][:,0]
            values_of_cell_row_boundary = values[cell_row_boundary_indices]
            _cell_layer = np.array((x_coordinates_of_cell_row_boundary, cell_row_boundary_indices.shape[0] * [y_coord_of_new_cell_layer])).T
            
            # add points left of boundary
            min_x_coord = np.min(x_coordinates_of_cell_row_boundary)
            x_coords_for_left_most = np.linspace(min_x_coord,min_x_coord-self.rectangle_length,5)
            index_of_left_most_point = np.where((self.BlockMesh_coordinates_2D[:,0] == min_x_coord) &
                                                (self.BlockMesh_coordinates_2D[:,1] == y_coord))[0]
            values_of_left_most_points = np.array(5* [values[index_of_left_most_point]]).reshape(5, values.shape[1])
            _cell_layer_left = np.array((x_coords_for_left_most, 5*[y_coord_of_new_cell_layer])).T
            # add points right of boundary
            max_x_coord = np.max(x_coordinates_of_cell_row_boundary)
            x_coords_for_right_most = np.linspace(max_x_coord,max_x_coord+self.rectangle_length,5)
            index_of_right_most_point = np.where((self.BlockMesh_coordinates_2D[:,0] == max_x_coord) &
                                                (self.BlockMesh_coordinates_2D[:,1] == y_coord))[0]
            values_of_right_most_points = np.array(5* [values[index_of_right_most_point]]).reshape(5, values.shape[1])
            _cell_layer_right = np.array((x_coords_for_right_most, 5*[y_coord_of_new_cell_layer])).T
            
            _cell_layer_new = np.vstack((_cell_layer_left, _cell_layer, _cell_layer_right)) 
            values_of_cell_row_boundary_new = np.vstack((values_of_left_most_points, values_of_cell_row_boundary, values_of_right_most_points)) 


            _cell_layers.append(_cell_layer_new)
            _cell_values.append(values_of_cell_row_boundary_new)

        _cell_values_array = np.array(_cell_values).reshape((n_new_layers*_cell_values[0].shape[0],_cell_values[0].shape[1]))
        _cell_values_list.append(_cell_values_array)

        # get left boundary (inlet)
        boundary_indices = np.where(self.BlockMesh_coordinates_2D[:,0] == np.min(self.BlockMesh_coordinates_2D[:,0]))[0]
        x_coord = np.min(self.BlockMesh_coordinates_2D[:,0])
        y_coordinates_of_boundary = self.BlockMesh_coordinates_2D[boundary_indices][:,1]
        values_of_boundary = values[boundary_indices]
        n_new_layers = 5
        for _x_coord in np.linspace(x_coord,x_coord-self.rectangle_length,n_new_layers+1)[1:]:
            _cell_layer = np.array((y_coordinates_of_boundary.shape[0] * [_x_coord], y_coordinates_of_boundary)).T
            _cell_layers.append(_cell_layer)
        _cell_values_array = np.tile(values_of_boundary, (n_new_layers,1)) #.reshape(1700,3)
        _cell_values_list.append(_cell_values_array)

        # get duct upper wall boundary (top)
        duct_indices = np.where(self.BlockMesh_coordinates_2D[:,0] <= 0)[0]  # do not stop at end of duct (at 0) because boundary of front facing wall would overlap, stop at -rectangle length 
        duct_upper_cells_y = np.max(self.BlockMesh_coordinates_2D[duct_indices,1])
        boundary_gap = 2*np.diff(np.sort(np.unique(self.BlockMesh_coordinates_2D[duct_indices,1]))[-2:]) # distance between last inner mesh cell and boundary condition cell shall be the distance between the two upper cells
        boundary_indices = np.nonzero(np.logical_and(
            (self.BlockMesh_coordinates_2D[:,1] == duct_upper_cells_y),
            self.BlockMesh_coordinates_2D[:,0] <= 0-self.rectangle_length))
        y_coord = np.min(self.BlockMesh_coordinates_2D[boundary_indices,1])
        x_coordinates_of_boundary = self.BlockMesh_coordinates_2D[boundary_indices][:,0]
        n_new_layers = 5
        # now add additional points left of the boundary (upstream of the inlet)
        x_coord = np.min(self.BlockMesh_coordinates_2D[:,0])
        additional_x_coordinates_left_of_boundary = np.linspace(x_coord,x_coord-self.rectangle_length,n_new_layers)[1:]
        x_coordinates_of_boundary = np.append(x_coordinates_of_boundary, additional_x_coordinates_left_of_boundary)
        final_shape_of_value_array = (x_coordinates_of_boundary.shape[0], values.shape[1])
        values_of_boundary = np.zeros(shape=final_shape_of_value_array)#values[boundary_indices]
        for _y_coord in np.linspace(y_coord+boundary_gap,y_coord+self.rectangle_length,n_new_layers):
            _cell_layer = np.array((x_coordinates_of_boundary, x_coordinates_of_boundary.shape[0] * [_y_coord])).T
            _cell_layers.append(_cell_layer)

        _cell_values_array = np.tile(values_of_boundary, (n_new_layers,1)) #.reshape(1700,3)
        _cell_values_list.append(_cell_values_array)

        # get front faces (fuel injector, secondary inlet and front wall)
        duct_height = 0.018
        above_duct_indices = np.where(self.BlockMesh_coordinates_2D[:,1] >= duct_height)
        x_coord = np.min(self.BlockMesh_coordinates_2D[above_duct_indices,0])
        min_y_coordinate_above_duct = np.min(self.BlockMesh_coordinates_2D[above_duct_indices,1]) # to skip the first y layer, get the first y coordinate that lies above the duct. 
        boundary_indices = np.nonzero(np.logical_and(
                (self.BlockMesh_coordinates_2D[:,0] == x_coord),
                (self.BlockMesh_coordinates_2D[:,1] > min_y_coordinate_above_duct)))
        y_coordinates_of_boundary = self.BlockMesh_coordinates_2D[boundary_indices][:,1]
        values_of_boundary = values[boundary_indices]
        n_new_layers = 5
        for _x_coord in np.linspace(x_coord,x_coord-self.rectangle_length,n_new_layers):
            _cell_layer = np.array((y_coordinates_of_boundary.shape[0] * [_x_coord], y_coordinates_of_boundary)).T
            _cell_layers.append(_cell_layer)
        _cell_values_array = np.tile(values_of_boundary, (n_new_layers,1)) #.reshape(1700,3)
        _cell_values_list.append(_cell_values_array)  

        # get top wall (surface on top)
        boundary_indices = np.where(self.BlockMesh_coordinates_2D[:,1] == np.max(self.BlockMesh_coordinates_2D[:,1]))[0]
        x_coordinates_of_boundary = self.BlockMesh_coordinates_2D[boundary_indices][:,0]
        top_wall_y_coord = np.min(self.BlockMesh_coordinates_2D[boundary_indices][:,1])
        values_of_boundary = np.zeros(shape=values[boundary_indices].shape)
        n_new_layers = 5
        for y_coord in np.linspace(top_wall_y_coord,top_wall_y_coord+self.rectangle_length,n_new_layers):
            _cell_layer = np.array((x_coordinates_of_boundary, x_coordinates_of_boundary.shape[0] * [y_coord])).T
            _cell_layers.append(_cell_layer)
        _cell_values_array = np.tile(values_of_boundary, (n_new_layers,1)) #.reshape(1700,3)
        _cell_values_list.append(_cell_values_array)

        # get outlet
        boundary_indices = np.where(self.BlockMesh_coordinates_2D[:,0] == np.max(self.BlockMesh_coordinates_2D[:,0]))[0]
        x_coord = np.max(self.BlockMesh_coordinates_2D[:,0])
        y_coordinates_of_boundary = self.BlockMesh_coordinates_2D[boundary_indices][:,1]
        values_of_boundary = values[boundary_indices]
        n_new_layers = 5
        for _x_coord in np.linspace(x_coord,x_coord+self.rectangle_length,n_new_layers):
            _cell_layer = np.array((y_coordinates_of_boundary.shape[0] * [_x_coord], y_coordinates_of_boundary)).T
            _cell_layers.append(_cell_layer)
        _cell_values_array = np.tile(values_of_boundary, (n_new_layers,1)) #.reshape(1700,3)
        _cell_values_list.append(_cell_values_array)

        # combine boundaries
        _cell_layers_combined = np.vstack(tuple(_cell_layers))
        _cell_values_combined = np.vstack(tuple(_cell_values_list))
        
        _x = np.vstack((self.BlockMesh_coordinates_2D, _cell_layers_combined))[:,0]
        _y = np.vstack((self.BlockMesh_coordinates_2D, _cell_layers_combined))[:,1]
        _c = np.vstack((values, _cell_values_combined))[:,0]

        ########################################################
        # plot all points
        if not self.cell_batches_plotted:
            sc_size = 0.001
            fig, axs = plt.subplots(figsize=(20,20))
            # plot scatter with original data
            sc = axs.scatter(_cell_layers_combined[:,0], _cell_layers_combined[:,1], c=_cell_values_combined[:,0], s=sc_size, cmap="RdBu_r")#, edgecolors='black')
            axs.set_aspect("equal")
            #axs.set_title("input data")
            axs.set_xlabel("x")
            axs.set_ylabel("y")

            y_bc_lines = [0, 0.018, 1.4]
            x_bc_lines = [-0.12, 0, 2.8]
            for _y in y_bc_lines:
                axs.plot(axs.get_xlim(), (_y, _y), c="k", linewidth=0.1)
            for _x in x_bc_lines:
                axs.plot((_x, _x), axs.get_ylim(), c="k", linewidth=0.1)
            # axs.set_xlim([-0.16, 0.10])
            # axs.set_ylim([-0.02, 0.05])
            #cbar1 = fig.colorbar(sc, ax=axs, location='right', anchor=(0.4, 0.4), shrink=0.6)#, ticks=ticks_list)
            axs.set_title("scatter")
            plt.tight_layout()
            #fig.savefig(os.path.join(self.path_to_postprocessing_folder, "scatter_test.png"), bbox_inches="tight", dpi=700)
            fig.savefig(os.path.join(self.path_to_postprocessing_folder, "scatter_test.svg"), bbox_inches="tight", format='svg', dpi=800)
            plt.close()

        # close_up_limits = {
        #     "x" : (-0.1,0.05),
        #     "y" : (0,0.1),
        # }
        # plot_scalar_field_on_mesh(_x, _y, 
        #             _c,f"boundary conditions",
        #             "x", "y", "values", f"boundary_condition_test",
        #             self.path_to_postprocessing_folder,
        #             close_up_limits=close_up_limits, aspect_ratio=None)
        
        print(f"applying boundary conditions took:  {(time.time()-_timer):.2f} seconds")

        return _cell_layers_combined, _cell_values_combined


    def create_interpolated_cell_batches(self, rectangle_length : float, points_per_batch_side : int):
        """
        """

        self.rectangle_length = rectangle_length
        self.points_per_batch_side = points_per_batch_side

        # the error is only defined on certain coordinate, to include the error as a value for interpolation it has to be enlarged to the full grid size
        last_error_projected_onto_full_grid = np.zeros(shape=(self.results["U"].shape[0],1))
        last_error_projected_onto_full_grid[self.coordinate_filter_indexes] = self.error_by_time[-1].reshape(self.error_by_time[-1].shape[0], 1)

        values = np.hstack((self.results["U"][:,0:2], last_error_projected_onto_full_grid))
        
        self.cell_batch_parmeter_names = ("Ux", "Uy", "error")

        x_range = [float(self.config["x_min"]), float(self.config["x_max"])]
        y_range = [float(self.config["y_min"]), float(self.config["y_max"])]

        batch_x_starting_points = np.arange(start=x_range[0], stop=x_range[1], step=self.rectangle_length)
        batch_y_starting_points = np.arange(start=y_range[0], stop=y_range[1], step=self.rectangle_length)

        theoretical_number_of_batches = len(batch_x_starting_points) * len(batch_y_starting_points)

        # TODO: This feature is needed at many positions ins the code, create function for entire class
        unique_x = np.unique(np.array(self.BlockMesh_coordinates_2D[:,0]))
        max_y_by_x = {}
        for x_value in unique_x:
            max_y_by_x[x_value] = np.max(self.BlockMesh_coordinates_2D[:,1][np.where(self.BlockMesh_coordinates_2D[:,0]==x_value)])


        def _point_is_in_mesh(x_point, y_point, max_y_by_x):
            y_interpolated = np.interp([x_point], list(max_y_by_x.keys()), list(max_y_by_x.values()))
            if y_point > y_interpolated:
                return False
            else:
                return True


        cell_batches = []
        cell_batch_counter = 0
        counter = 0
        _cellbatch_loop_start = time.time()
        for ix in range(len(batch_x_starting_points)-1):
            for iy in range(len(batch_y_starting_points)-1):

                if _point_is_in_mesh(batch_x_starting_points[ix], batch_y_starting_points[iy], max_y_by_x):
                    counter += 1
                    if counter % 100 == 0:
                        self.logger.info(f"checking batches: {counter}/{theoretical_number_of_batches} x:[{batch_x_starting_points[ix]:.3f}, {batch_x_starting_points[ix+1]:.3f}], y:[{batch_y_starting_points[iy]:.3f}, {batch_y_starting_points[iy+1]:.3f}]") 
                    _st = time.time()
                    x_batch = np.linspace(batch_x_starting_points[ix], batch_x_starting_points[ix+1], num=self.points_per_batch_side)
                    y_batch = np.linspace(batch_y_starting_points[iy], batch_y_starting_points[iy+1], num=self.points_per_batch_side)
                    
                    point_inside_x = np.logical_and((self.BlockMesh_coordinates_2D[:,0] >= np.min(x_batch)), (self.BlockMesh_coordinates_2D[:,0] <= np.max(x_batch)))
                    point_inside_y = np.logical_and((self.BlockMesh_coordinates_2D[:,1] >= np.min(y_batch)), (self.BlockMesh_coordinates_2D[:,1] <= np.max(y_batch)))
                    number_of_points_in_cell_batch = np.sum(np.logical_and(point_inside_x, point_inside_y))
                    duration = time.time() - _st
                    #self.logger.info(f"time: {duration:.5f}")
                    if number_of_points_in_cell_batch > 1:
                        xx_batch, yy_batch = np.meshgrid(x_batch, y_batch)
                        _st = time.time()
                        # now extract mesh points from original grid including some extra space around it to enable interpolation but reduce number of grid point passed to griddata function
                        pad = 30*self.rectangle_length
                        point_inside_x = np.logical_and((self.BlockMesh_coordinates_2D[:,0] >= np.min(x_batch)-pad), (self.BlockMesh_coordinates_2D[:,0] <= np.max(x_batch)+pad))
                        point_inside_y = np.logical_and((self.BlockMesh_coordinates_2D[:,1] >= np.min(y_batch)-pad), (self.BlockMesh_coordinates_2D[:,1] <= np.max(y_batch)+pad))
                        point_in_cell_batch = np.logical_and(point_inside_x, point_inside_y)
                        cropped_grid = self.BlockMesh_coordinates_2D[point_in_cell_batch]
                        cropped_data = values[point_in_cell_batch]
                        # compute interpolated grid 
                        grid_z1 = griddata(cropped_grid, cropped_data, (xx_batch, yy_batch), method='nearest')  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
                        duration = time.time() - _st
                        #self.logger.info(f"grid creation: {duration:.5f}")
                        array_has_nan = np.isnan(np.sum(grid_z1))
                        # if everey entry is a valid number, proceed to append the cell batch and plot the data
                        if not array_has_nan:
                            self.logger.info(f"creating cell batch {cell_batch_counter}")
                            cell_batches.append((x_batch, y_batch, grid_z1))
                            # plot every 300th created cell batch
                            if (cell_batch_counter % 300) == 0 and not self.cell_batches_plotted:
                                plot_interpolated_cell_batch(grid_z1, x_batch, y_batch,
                                                self.BlockMesh_coordinates_2D, values, 
                                                "comparison of cell batch interpolation to original data",
                                                f"cell_batch_{cell_batch_counter}.png",
                                                self.path_to_postprocessing_folder, self.logger)

                            cell_batch_counter += 1
                        else:
                            grid_zero = np.zeros(shape=grid_z1.shape)
                            plot_interpolated_cell_batch(grid_zero, x_batch, y_batch,
                                            self.BlockMesh_coordinates_2D, values, 
                                            "comparison of cell batch interpolation to original data",
                                            f"cell_batch_failed_{counter}.png",
                                            self.path_to_postprocessing_folder, self.logger)
                    else:
                        print(number_of_points_in_cell_batch)
                        # grid_zero = np.zeros(shape=grid_z1.shape)
                        # plot_interpolated_cell_batch(grid_zero, x_batch, y_batch,
                        #                     self.BlockMesh_coordinates_2D, values, 
                        #                     "comparison of cell batch interpolation to original data",
                        #                     f"cell_batch_failed_{counter}.png",
                        #                     self.path_to_postprocessing_folder, self.logger)
                    

        _cellbatch_loop_end = time.time()
        _cellbatch_loop_duration = _cellbatch_loop_end - _cellbatch_loop_start
        self.logger.info(f"TIMER: cell batch creation loop took: {_cellbatch_loop_duration:.3f} seconds")

        if not self.cell_batches_plotted:
            plot_cell_batch_coverage(cell_batches, self.BlockMesh_coordinates_2D, values, "cell batch coverage",
                                                f"cell_batch_coverage.png",
                                                self.path_to_postprocessing_folder, self.logger)

        self.cell_batches = cell_batches
        self.cell_batches_plotted = True  # at object init this variabel is set to false. Therefore the plotting will only be done at the first call of this method

        self.logger.info(f"finished cell batch creation")


    def interpolate_from_cell_batches_onto_original_grid(self, action_batches):
        """
        """
        self.logger.info("interpolate_from_cell_batches_onto_original_grid")

        close_up_limits = {
            "x" : (-.15, 0.2),
            "y" : (0, 0.05)
        }

        from scipy import interpolate
        # construct large grid from batches
        self.actions_on_grid = np.zeros(shape=self.BlockMesh_coordinates_2D[:,0].shape)

        for cell_batch, action_batch, counter in zip(self.cell_batches, action_batches, range(len(action_batches))):

            if action_batch.shape == (1,1):
                action_batch = action_batch * np.ones(shape=(cell_batch[0].shape[0], cell_batch[1].shape[0]))
            
            # loop through all batches
            # if counter % 100 == 0:
            #     self.logger.info(f"interpolating action {counter}")

            #     plot_scalar_field_on_mesh(self.BlockMesh_coordinates_2D[:,0], self.BlockMesh_coordinates_2D[:,1],
            #         self.actions_on_grid,
            #         f"interpolation",
            #         "x", "y", "", f"interpolation_of_action_onto_mesh_{counter}",
            #         self.path_to_postprocessing_folder,

            #         close_up_limits=close_up_limits, cmap="jet", center_cbar=False,  aspect_ratio=1)

            x_batch = cell_batch[0]
            y_batch = cell_batch[1]
            xx_batch, yy_batch = np.meshgrid(x_batch, y_batch)

            # now find original grid points that are inside the current batch
            pad = 0
            point_inside_x = np.logical_and((self.BlockMesh_coordinates_2D[:,0] >= np.min(x_batch)-pad), (self.BlockMesh_coordinates_2D[:,0] <= np.max(x_batch)+pad))
            point_inside_y = np.logical_and((self.BlockMesh_coordinates_2D[:,1] >= np.min(y_batch)-pad), (self.BlockMesh_coordinates_2D[:,1] <= np.max(y_batch)+pad))
            point_in_cell_batch = np.logical_and(point_inside_x, point_inside_y)
            points_in_orig_grid = self.BlockMesh_coordinates_2D[point_in_cell_batch]

            # So ist das viel zu langsam! 
            f = interpolate.interp2d(xx_batch, yy_batch, action_batch, kind='linear')
            self.actions_on_grid[point_in_cell_batch.nonzero()] = [f(_x, _y)[0] for _x, _y in zip(points_in_orig_grid[:,0],points_in_orig_grid[:,1])]

            #znew = np.array([f(_x, _y) for _x, _y in zip(points_in_orig_grid[:,0],points_in_orig_grid[:,1])])
            # znew += znew
            # Ich muss das so machen: action_batch ist momentan (5,5) groß, Ich muss die innere xy sortierung irgendwie herausfinden und dann x:(25,0), y:(25,0) action:(25,0) ermitteln. 
            # # IDEE:
            # x_1d = xx_batch.flatten()
            # y_1d = yy_batch.flatten()
            # action_batch_1d = action_batch.flatten()
            # # dann alle appenden
            # x_1d_all_batches = np.concatenate((x_1d_all_batches, x_1d))
            # y_1d_all_batches = np.concatenate((y_1d_all_batches, y_1d))
            # action_batches_1d = np.concatenate((action_batches_1d, action_batch_1d))


        # diese vektoren muss ich dann aneinander hängen, dann habe ich drei vektoren a (1520*25,), die kann ich dann nehmen, um eine interpolation zu machen
        # end_idx = 5000
        # _start = time.time()
        # f = interpolate.interp2d(x_1d_all_batches[0:end_idx], y_1d_all_batches[0:end_idx], action_batches_1d[0:end_idx], kind='linear')
        # _interp_time = time.time() - _start 
        # self.logger.info(f"interpolation onto mesh took: {_interp_time:.3f} seconds")

        plot_scalar_field_on_mesh(self.BlockMesh_coordinates_2D[:,0], self.BlockMesh_coordinates_2D[:,1],
                            self.actions_on_grid,
                            f"interpolation",
                            "x", "y", "", f"interpolation_of_action_onto_mesh_t{self.time_list[-1]}",
                            self.path_to_postprocessing_folder,
                            close_up_limits=close_up_limits, cmap="jet", center_cbar=False,  aspect_ratio=1)



    def split_cells_into_batches_clustering(self, n_batches=20, cells_per_batch : int = None) -> tuple: 
        """
        """
        # cluster variant

        self.logger.info("now splitting cells into batches using clustering method")
        self.n_cells = self.BlockMesh_coordinates_2D.shape[0]
        self.n_cell_batches = n_batches

        # kmeans
        kmeans = KMeans(n_clusters=n_batches, random_state=0).fit(self.BlockMesh_coordinates_2D)
        labels = kmeans.labels_
        self.batch_id_per_cell = kmeans.predict(self.BlockMesh_coordinates_2D)
        self.batch_index_list = [np.where(labels == l) for l in range(n_batches)]
        self.batch_x_coordinates = [self.BlockMesh_coordinates_2D[batch_indexes,0] for batch_indexes in self.batch_index_list]
        self.batch_y_coordinates = [self.BlockMesh_coordinates_2D[batch_indexes,1] for batch_indexes in self.batch_index_list]

        plot_scalar_field_on_mesh(self.BlockMesh_coordinates_2D[:,0], self.BlockMesh_coordinates_2D[:,1],
                    self.batch_id_per_cell,
                    f"mesh cell batches - batches",
                    "x", "y", "batch id", f"cell_batches_clustering_method",
                    self.path_to_postprocessing_folder,
                    close_up_limits=None, cmap="gist_rainbow", center_cbar=False, aspect_ratio=1,
                    custom_axis_limits=None, levels_overwrite=n_batches)

        if n_batches < 20:
            for batch_number, (x_coord, y_coord) in enumerate(zip(self.batch_x_coordinates, self.batch_y_coordinates)):
                cell_ids = np.arange(x_coord.shape[1])
                plot_scalar_field_on_mesh(x_coord.T, y_coord.T,
                cell_ids,
                f"mesh cell batch no {batch_number}",
                "x", "y", "cell id", f"cell_batch_{batch_number}_cells_clustering_method",
                self.path_to_postprocessing_folder,
                close_up_limits=None, cmap="gist_rainbow", center_cbar=False, aspect_ratio=1,
                custom_axis_limits=None, levels_overwrite=n_batches)
        else:
            # intra batch cell order
            for batch_id in np.linspace(start=0, stop=n_batches-1, num=3, dtype=int):
                _x_list = self.batch_x_coordinates[batch_id][:].T
                _y_list = self.batch_y_coordinates[batch_id][:].T
                fig = plt.figure()
                for cell_id, (_x, _y) in enumerate(zip(_x_list, _y_list)):
                    plt.scatter(_x, _y, label=str(cell_id), s=20)
                    plt.text(_x, _y, s=str(cell_id))
                plt.xlabel("x")
                plt.ylabel("y")
                plt.title(f"batch array indexes of batch {batch_id}")
                plt.savefig(f"{self.path_to_postprocessing_folder}/batch_{batch_id}_clustering_method.png", dpi=300)
                plt.close()


        return self.batch_index_list, self.batch_x_coordinates, self.batch_y_coordinates
    
    def process_actions(self, action_batches : list):

        if len(action_batches) == len(self.cell_ids): 
    
            return np.array(action_batches).reshape((len(action_batches),1))
        
        else:
            # interpolate

            triangulation = Delaunay(self.coordinates_of_selected_cell_ids)  # Compute the triangulation
            data_of_actions = np.array(action_batches).reshape((len(action_batches),1))
            interpolator = LinearNDInterpolator(triangulation, data_of_actions)
            interpolated_actions_on_full_grid = interpolator(self.BlockMesh_coordinates_2D)

            if True in np.isnan(interpolated_actions_on_full_grid):
                self.logger.warning("nan is in interpolation data")

            interpolated_actions_on_full_grid = np.nan_to_num(interpolated_actions_on_full_grid, nan=0, posinf=0)

            plot_scalar_field_on_mesh(self.BlockMesh_coordinates_2D[:,0], self.BlockMesh_coordinates_2D[:,1],
                        interpolated_actions_on_full_grid,
                        f"interpolated_actions_on_full_grid",
                        "x", "y", "batch id", f"interpolated_actions_on_full_grid",
                        self.path_to_postprocessing_folder,
                        close_up_limits=None, cmap="gist_rainbow", center_cbar=False, aspect_ratio=1,
                        custom_axis_limits=None)

            return interpolated_actions_on_full_grid
