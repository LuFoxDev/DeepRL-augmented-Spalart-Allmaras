from logging import Logger
from modules.config import Config
from modules.utilities.terminal import run_command
from modules.utilities.file_helpers import modify_file
import os
from typing import Union
import filecmp
from shutil import copyfile, copytree


class OpenFoamCore():
    """
    This class establishes a connection to the OpenFoam source code and the methods can be used to modify code and recompile it.
    """

    def __init__(self, logger : Logger, config : Config) -> None:

        self.logger = logger
        self.config = config

        self.path_to_OF_core = os.path.join(self.config["path_to_openfoam_core"])

        self.path_to_openfoamCore_templates = os.path.join(
             "./templates", "openfoamCore", self.config["name_of_open_foam_core_template"])
        
        self.openFoamCoreDestination = self.config["path_to_openfoam_turbulence_model"]


    def change_custom_SpalartAllmaras_scalars(self, change_dict : Union[list, dict] = {}) -> None:
        """
        opens template source code files, changes the customs scalars according to the dictionary
        saves the files and copies them to the target openfoam core directory
        """

        if type(change_dict) == list:
            placeholders = [f"CUSTOM_SCALAR_{number}" for number in range(1, len(change_dict))]
            replacements = change_dict
        else:            
            placeholders = change_dict.keys()
            replacements = change_dict.values()

        self.logger.info("now changing CUSTOM_SCALAR in the Spalart-Allmaras files turbulence models")
        self.logger.info(f"CUSTOM_SCALAR values: {replacements}")
        # open template files and hange according to dict and save files to target dir
        for fname in ["SpalartAllmaras.C", "SpalartAllmaras.H"]:
            source_file = os.path.join(self.path_to_openfoamCore_templates, fname)
            destination_file = os.path.join(self.path_to_OF_core, "src/TurbulenceModels/turbulenceModels/RAS/SpalartAllmaras/", fname)

            modify_file(source_file, destination_file, placeholders, replacements)

        self.logger.info("now recompiling turbulence models")
        # preface = "source $HOME/OpenFOAM/OpenFOAM-4.x/etc/bashrc WM_LABEL_SIZE=64 FOAMY_HEX_MESH=yes"
        # command = "/home/lukas/OpenFOAM/OpenFOAM-4.x/src/TurbulenceModels/Allwmake"
        # run_command(preface + "\n" + command)
        path_to_script = "scripts/recompile_TurbulenceModels.sh"
        command = f"bash {path_to_script}"
        results = run_command(command)
        self.logger.info("recompiled turbulence models")

    def compile_turbulence_model(self, force_compilation=False):
        """
        first check if files from template folder were already copied to target folder. If not copy the file
        Compile turbulence model (unless files were already up to date)
        """
        self.logger.info("checking src files")
        self.compilation_pending = False
        # compare openfoam source with template files
        for fname in os.listdir(self.path_to_openfoamCore_templates):
            source_file = os.path.join(self.path_to_openfoamCore_templates, fname)
            destination_file = os.path.join(self.path_to_OF_core, self.openFoamCoreDestination, fname)
            
            if os.path.exists(destination_file):
                # the destination file or path exists, now check if the content equals
                if os.path.isdir(source_file) or os.path.isdir(destination_file):
                    # if it is a directory check all subfiles
                    files_equal = True
                    for source_fname in os.listdir(source_file):
                        _source_file = os.path.join(source_file, source_fname)
                        _destination_file = os.path.join(destination_file, source_fname)
                        files_equal = filecmp.cmp(_source_file, _destination_file)
                else: 
                    # if it is just a file, compare the files
                    files_equal = filecmp.cmp(source_file, destination_file)
            else:
                # if the destination file does not exist
                self.logger.info(f"{fname} in openfoam source folder was not yet created")
                files_equal = False
            
            if files_equal:
                self.logger.info(f"{fname} in openfoam source folder was already copied from template folder")
            
            else:
                self.compilation_pending = True
                if os.path.isdir(source_file) or os.path.isdir(destination_file):
                    copytree(source_file, destination_file, dirs_exist_ok=True)
                else:
                    copyfile(source_file, destination_file)
                self.logger.info(f"copied {fname} from templates to {self.openFoamCoreDestination}")

        # compile turbulence model
        if self.compilation_pending or force_compilation:
            self.logger.info(f"recompiling OpenFoam Turbulence Models")
            path_to_script = "scripts/recompile_TurbulenceModels.sh"
            command = f"bash {path_to_script}"
            results = run_command(command)
            self.logger.info("recompiled turbulence models")
        else:
            self.logger.info(f"did not recompile OpenFoam Turbulence Models because files were already copire to target folder")




    
    

