# import of own classes and functions
from logging import Logger
from typing import Tuple
from modules.config import Config
import modules.utilities.FF_funs as ff
from modules.plotting.plotting_functions import plot_coordinates, plot_scalar_field_on_mesh, plot_inlet_velocity
from modules.openfoam.extract_inlet_velocity import extract_inlet_velocity 

# import of general modules
import os
import numpy as np
from scipy import interpolate
import meshio
import pickle
#from shutil import copyfile, copytree
#import subprocess
#import traceback
#from pathlib import Path
#import stat

from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib as mlp
mlp.use("agg", force=True) # mlp.use("agg")
from matplotlib import colors

    


class LESresultsWorker():

    def __init__(self, logger : Logger, config : Config) -> None:
        
        self.logger = logger
        self.config = config
        
        self.path_to_export_folder_for_this_run = os.path.join(
            self.config["export_folder"], self.config["name_of_run"])
        
        self.path_to_postprocessing_folder = os.path.join(
            self.config["export_folder"], self.config["name_of_run"], "post-processing", "")

        self.path_to_LES_results = self.config["path_to_les_results"]
        self.inlet_x_coodinate = self.config["inlet_x_coodinate"]

        if not os.path.exists(self.path_to_postprocessing_folder):
            os.makedirs(self.path_to_postprocessing_folder)

    def _z2polar(self, z): 
        return (abs(z), np.angle(z))
    
    def load_data(self) -> None:
        """
        imports LES vtk file
        """

        # Import vtk LES jetflow data
        self.logger.info(f"read mesh: {self.path_to_LES_results}")
        mesh = meshio.read(self.path_to_LES_results)
        self.mesh = mesh
        self.logger.info(f"finished importing mesh")

        U = np.array(mesh.point_data['UMean'])
        U_Gradient = np.array(mesh.point_data['UMean_Gradient'])
        self.logger.info(f"Umean data shape: {U.shape}")
        ux = U[:, 0]
        uy = U[:, 1]
        uz = U[:, 2]
        direction_names = [f"delU{i}_del{j}" for i in ["x","y","z"] for j in ["x","y","z"]]
        self.U_mean_grad_dict = dict(zip(direction_names, U_Gradient.T))
        coordinates = mesh.points
        U_p = np.array(mesh.point_data['UPrime2Mean'])  # these are the R-stresses
        # define coordinates
        coo_LES = np.array(mesh.points)
        uxx = U_p[:, 0]
        uyy = U_p[:, 1]
        uzz = U_p[:, 2]
        uxy = U_p[:, 3]
        uyz = U_p[:, 4]
        uxz = U_p[:, 5]
        x = coo_LES[:, 0]
        y = coo_LES[:, 1]
        z = coo_LES[:, 2]

        self.coordinates = coo_LES

        # if self.config["make_plots"]:
        #     if not os.path.exists(self.path_to_postprocessing_folder):
        #         os.makedirs(self.path_to_postprocessing_folder)
        #     plot_coordinates(x, y, z, source="LES data",
        #                     export_path=self.path_to_postprocessing_folder)

        # coordinate transform
        Z = y + 1j * z

        rho, theta = self._z2polar(Z)
        # define polar coordinates x, rho, theta
        coo_old = np.zeros((len(rho), 3))
        coo_old[:, 0] = x
        coo_old[:, 1] = rho
        coo_old[:, 2] = theta

        self.coordinates_dict = {
            "x" : x,
            "y" : y,
            "z" : z,  
            "rho" : rho,
            "theha" : theta
        }

        #######################################################################
        # Calculate velocities
        # ux = ux  # no changes here
        ur = np.cos(theta)*uy + np.sin(theta)*uz
        ut = -np.sin(theta)*uy + np.cos(theta)*uz
        urr = np.cos(theta)**2*uyy + np.sin(theta)**2 * \
            uzz + 2*np.sin(theta)*np.cos(theta)*uyz
        utt = np.cos(theta)**2*uzz + np.sin(theta)**2 * \
            uyy - 2*np.sin(theta)*np.cos(theta)*uyz
        uxr = np.cos(theta)*uxy + np.sin(theta)*uxz
        uxt = -np.sin(theta)*uxy + np.cos(theta)*uxz
        urt = np.cos(theta)**2*uyz - np.sin(theta)**2*uyz + np.sin(theta) * \
            np.cos(theta)*uzz - np.sin(theta)*np.cos(theta)*uyy
        
        delUx_delr = np.cos(theta)*self.U_mean_grad_dict["delUx_dely"] + np.sin(theta)*self.U_mean_grad_dict["delUx_delz"]
        delUr_delx = np.cos(theta)*self.U_mean_grad_dict["delUy_delx"] + np.sin(theta)*self.U_mean_grad_dict["delUz_delx"]
        delUr_delr = np.cos(theta)*self.U_mean_grad_dict["delUy_dely"] + np.sin(theta)*self.U_mean_grad_dict["delUz_delz"]

        self.velocities = {
            "ux" : ux,
            "uy" : uy,
            "uz" : uz,
            "uxx" : uxx,
            "uyy" : uyy,
            "uzz" : uzz,
            "uxy" : uxy,
            "uyz" : uyz,
            "uxz" : uxz,
            "ur" : ur,
            "ut" : ut,
            "urr" : urr,
            "utt" : utt,
            "uxr" : uxr,
            "uxt" : uxt,
            "urt" : urt,
            "delUx_delr" : delUx_delr,
            "delUr_delx" : delUr_delx,
            "delUr_delr" : delUr_delr,
            "delUx_delx" : self.U_mean_grad_dict["delUx_delx"],
            "delUx_dely" : self.U_mean_grad_dict["delUx_dely"],
            "delUx_delz" : self.U_mean_grad_dict["delUx_delz"],
            "delUy_delx" : self.U_mean_grad_dict["delUy_delx"],
            "delUz_delx" : self.U_mean_grad_dict["delUz_delx"]
        }


    def perform_azimutal_averaging(self, BlockMesh_coordinates_2D : np.array) -> None:
        """
        perform azimutal averaging of 3D data to get axisymmetric 2D data
        """
        #############################################################################
        # extend in azimuthal dimension
        self.logger.info("coo_blockMesh_2D_exp_car = ff.ExpandForAzimuthalAverage(coo_blockMesh_2D,100)")
        coo_blockMesh_2D_exp_car = ff.ExpandForAzimuthalAverage(BlockMesh_coordinates_2D, 100)
        # if self.config["make_plots"]:
        #     plot_coordinates(coo_blockMesh_2D_exp_car[:, 0], coo_blockMesh_2D_exp_car[:, 1],
        #                     coo_blockMesh_2D_exp_car[:, 2], source="coo_blockMesh_2D_exp_car", export_path=self.path_to_postprocessing_folder)
            
        # this is still in x y z
        coo_blockMesh_2D_exp = np.zeros((len(coo_blockMesh_2D_exp_car[:, 1]), 3))
        # coor trafo
        Z = coo_blockMesh_2D_exp_car[:, 1] + 1j * coo_blockMesh_2D_exp_car[:, 2]
        rho_exp, theta_exp = self._z2polar(Z)
        coo_blockMesh_2D_exp[:, 0] = coo_blockMesh_2D_exp_car[:, 0]
        coo_blockMesh_2D_exp[:, 1] = rho_exp
        coo_blockMesh_2D_exp[:, 2] = theta_exp

        # stack variables for interpolation
        field_count = len(self.velocities)
        # temp = np.zeros((len(self.velocities["ux"]), field_count))
        temp = np.array(list(self.velocities.values())).T

        # interpolate
        # first linear
        # temp_exp = np.squeeze(interpolate.griddata(self.coordinates, temp, coo_blockMesh_2D_exp_car, method='linear'))
        # second nearest for the nan points
        self.logger.info("now interpolating (nearest)")
        temp_exp = np.squeeze(interpolate.griddata(
            self.coordinates, temp, coo_blockMesh_2D_exp_car, method='nearest'))
        
        # temp_exp[np.isnan(temp_exp)]=temp_exp2[np.isnan(temp_exp)]
        # azimuthal averaging (this is again on cartesian coordinates)
        self.logger.info("azimuthal averaging")
        mean_field_2D = np.zeros((len(BlockMesh_coordinates_2D[:,0]), field_count))
        for ind in range(field_count):
            mean_field_2D[:, ind] = ff.ContractFromAximuthalAverage(
                BlockMesh_coordinates_2D, 100, temp_exp[:, ind])

        self.mean_field = mean_field_2D
        self.mean_fields_dict = {}
        for ind, key in enumerate(self.velocities.keys()):
            self.mean_fields_dict[key] = mean_field_2D[:, ind]

        # {'ux': mean_field_2D[:, 0], 'ur': mean_field_2D[:, 1], 'ut': mean_field_2D[:, 2],
        #                         'uxx': mean_field_2D[:, 3], 'urr': mean_field_2D[:, 4], 'utt': mean_field_2D[:, 5],
        #                         'uxr': mean_field_2D[:, 6], 'uxt': mean_field_2D[:, 7], 'urt': mean_field_2D[:, 8],
        #                         'delUx_delr' : mean_field_2D[:, 9]}
        self.logger.info("now dumping mean fields")

        if not os.path.exists(self.path_to_postprocessing_folder):
            os.makedirs(self.path_to_postprocessing_folder)

        # D = 0.035
        # close_up_limits = {
        #     "x" : [-0.15/D, 1/D],
        #     "y" : [0/D, 0.2/D]}
        # from modules.plotting.plotting_functions import plot_scalar_field_with_streamlines_on_mesh
        # for key in self.mean_fields_dict.keys():
        #     plot_scalar_field_with_streamlines_on_mesh(
        #         BlockMesh_coordinates_2D[:,0]/D,
        #         BlockMesh_coordinates_2D[:,1]/D,
        #         self.mean_fields_dict["ux"],
        #         self.mean_fields_dict["ur"],
        #         self.mean_fields_dict[key],
        #         " ",  ##f"{field_name}\n{name}",
        #         "x/D", "r/D", key, f"{key}_withStreamline",
        #         self.path_to_postprocessing_folder,
        #         close_up_limits=close_up_limits, cmap="jet", center_cbar=False,
        #         aspect_ratio=1, close_up_aspect_ratio=3, 
        #         streamline_color="black")#

        pickle.dump(self.mean_fields_dict,
                open(os.path.join(self.path_to_postprocessing_folder, "meanfields.p"), "wb"))

        # --------------------------------------------------
        # plot velocity fields o mean field

        # close_up_limits = {
        #     "x" : (float(self.config["close_up_limits_x_lower"]), float(self.config["close_up_limits_x_upper"])),
        #     "y" : (float(self.config["close_up_limits_y_lower"]), float(self.config["close_up_limits_y_upper"]))
        # }

        # import matplotlib as mlp
        # import matplotlib.pyplot as plt
        # mlp.use("agg", force=True) # mlp.use("agg")
        # for name, field in self.mean_fields_dict.items():
        #     c = field.reshape(BlockMesh_coordinates_2D[:,0].shape)
        #     ######################################
        #     plt.close()
        #     plt.figure(figsize=(4,4))
        #     x = BlockMesh_coordinates_2D[:,0]
        #     y = BlockMesh_coordinates_2D[:,1]
        #     colors = c
        #     sc=plt.scatter(x, y, s=2, c=colors, alpha=1.0)
        #     plt.colorbar(sc)
        #     plt.tight_layout()
        #     plt.savefig(f"{self.path_to_postprocessing_folder}//mean_field_{name}.png")

        #     ###########################################
        #     z_margin = 0.001
        #     x_min = -0.1
        #     x_max = 1
        #     y_min = 0
        #     y_max = 0.2
        #     coordinate_filter_for_close_up_limits = np.where((self.coordinates[:,0] >= x_min)\
        #                                         & (self.coordinates[:,0] <= x_max)\
        #                                         & (self.coordinates[:,1] >= y_min)\
        #                                         & (self.coordinates[:,1] <= y_max)\
        #                                         & (self.coordinates[:,2] >= -z_margin)\
        #                                         & (self.coordinates[:,2] <= z_margin))[0]
        #     coordinates_sliced = self.coordinates[coordinate_filter_for_close_up_limits]
        #     print(f"{coordinates_sliced.shape[0]/self.coordinates.shape[0]*100:.1f} percent")
        #     plt.close()
        #     plt.figure()
        #     x = coordinates_sliced[:,0]
        #     y = coordinates_sliced[:,1]
        #     colors = self.velocities[name][coordinate_filter_for_close_up_limits]
        #     sc=plt.scatter(x, y, s=1, c=colors, alpha=0.5)
        #     plt.colorbar(sc)
        #     plt.savefig(f"{self.path_to_postprocessing_folder}//{name}_sliced.png")

    def perform_azimutal_averaging_new(self, BlockMesh_coordinates_2D : np.array) -> None:
        """
        perform azimutal averaging of 3D data to get axisymmetric 2D data
        """
        #############################################################################
        # extend in azimuthal dimension
        self.logger.info("coo_blockMesh_2D_exp_car = ff.ExpandForAzimuthalAverage(coo_blockMesh_2D,100)")
        coo_blockMesh_2D_exp_car = ff.ExpandForAzimuthalAverage(BlockMesh_coordinates_2D, 100)
        # if self.config["make_plots"]:
        #     plot_coordinates(coo_blockMesh_2D_exp_car[:, 0], coo_blockMesh_2D_exp_car[:, 1],
        #                     coo_blockMesh_2D_exp_car[:, 2], source="coo_blockMesh_2D_exp_car", export_path=self.path_to_postprocessing_folder)
            
        # this is still in x y z
        coo_blockMesh_2D_exp = np.zeros((len(coo_blockMesh_2D_exp_car[:, 1]), 3))
        # coor trafo
        Z = coo_blockMesh_2D_exp_car[:, 1] + 1j * coo_blockMesh_2D_exp_car[:, 2]
        rho_exp, theta_exp = self._z2polar(Z)
        coo_blockMesh_2D_exp[:, 0] = coo_blockMesh_2D_exp_car[:, 0]
        coo_blockMesh_2D_exp[:, 1] = rho_exp
        coo_blockMesh_2D_exp[:, 2] = theta_exp

        # stack variables for interpolation
        temp = np.zeros((len(self.velocities["ux"]), 9))
        temp[:, 0] = self.velocities["ux"]
        temp[:, 1] = self.velocities["ur"]
        temp[:, 2] = self.velocities["ut"]
        temp[:, 3] = self.velocities["uxx"]
        temp[:, 4] = self.velocities["urr"]
        temp[:, 5] = self.velocities["utt"]
        temp[:, 6] = self.velocities["uxr"]
        temp[:, 7] = self.velocities["uxt"]
        temp[:, 8] = self.velocities["urt"]

        # interpolate
        # first linear
        # temp_exp = np.squeeze(interpolate.griddata(self.coordinates, temp, coo_blockMesh_2D_exp_car, method='linear'))
        # second nearest for the nan points
        self.logger.info("now interpolating (nearest)")
        temp_exp = np.squeeze(interpolate.griddata(
            self.coordinates, temp, coo_blockMesh_2D_exp_car, method='nearest'))
        # temp_exp[np.isnan(temp_exp)]=temp_exp2[np.isnan(temp_exp)]

        # azimuthal averaging (this is again on cartesian coordinates)
        self.logger.info("azimuthal averaging")
        mean_field_2D = np.zeros((len(BlockMesh_coordinates_2D[:,0]), 9))
        for ind in range(9):
            mean_field_2D[:, ind] = ff.ContractFromAximuthalAverage(
                BlockMesh_coordinates_2D, 100, temp_exp[:, ind])

        self.mean_field = mean_field_2D
        self.mean_fields_dict = {'ux': mean_field_2D[:, 0], 'ur': mean_field_2D[:, 1], 'ut': mean_field_2D[:, 2], 'uxx': mean_field_2D[:, 3],
                    'urr': mean_field_2D[:, 4], 'utt': mean_field_2D[:, 5], 'uxr': mean_field_2D[:, 6], 'uxt': mean_field_2D[:, 7], 'urt': mean_field_2D[:, 8]}
        self.logger.info("now dumping mean fields")

        if not os.path.exists(self.path_to_postprocessing_folder):
            os.makedirs(self.path_to_postprocessing_folder)

        pickle.dump(self.mean_fields_dict,
                open(os.path.join(self.path_to_postprocessing_folder, "meanfields.p"), "wb"))

        # --------------------------------------------------
        # plot velocity fields o mean field

        close_up_limits = {
            "x" : (float(self.config["close_up_limits_x_lower"]), float(self.config["close_up_limits_x_upper"])),
            "y" : (float(self.config["close_up_limits_y_lower"]), float(self.config["close_up_limits_y_upper"]))
        }

        import matplotlib as mlp
        import matplotlib.pyplot as plt
        mlp.use("agg", force=True) # mlp.use("agg")
        for name, field in self.mean_fields_dict.items():
            c = field.reshape(BlockMesh_coordinates_2D[:,0].shape)
            ######################################
            plt.close()
            plt.figure(figsize=(4,4))
            x = BlockMesh_coordinates_2D[:,0]
            y = BlockMesh_coordinates_2D[:,1]
            colors = c
            sc=plt.scatter(x, y, s=2, c=colors, alpha=1.0)
            plt.colorbar(sc)
            plt.tight_layout()
            plt.savefig(f"{self.path_to_postprocessing_folder}//mean_field_{name}.png")

            ###########################################
            z_margin = 0.001
            x_min = -0.1
            x_max = 1
            y_min = 0
            y_max = 0.2
            coordinate_filter_for_close_up_limits = np.where((self.coordinates[:,0] >= x_min)\
                                                & (self.coordinates[:,0] <= x_max)\
                                                & (self.coordinates[:,1] >= y_min)\
                                                & (self.coordinates[:,1] <= y_max)\
                                                & (self.coordinates[:,2] >= -z_margin)\
                                                & (self.coordinates[:,2] <= z_margin))[0]
            coordinates_sliced = self.coordinates[coordinate_filter_for_close_up_limits]
            print(f"{coordinates_sliced.shape[0]/self.coordinates.shape[0]*100:.1f} percent")
            plt.close()
            plt.figure()
            x = coordinates_sliced[:,0]
            y = coordinates_sliced[:,1]
            colors = self.velocities[name][coordinate_filter_for_close_up_limits]
            sc=plt.scatter(x, y, s=1, c=colors, alpha=0.5)
            plt.colorbar(sc)
            plt.savefig(f"{self.path_to_postprocessing_folder}//{name}_sliced.png")

    def extract_inlet_velocity(self, BlockMesh_coordinates_2D : np.array) -> Tuple:
        """
        """
        self.logger.info("now extracting inlet position from LES mean field data")

        # parameter definition
        inlet_position = float(self.config["inlet_x_coodinate"])
        x = BlockMesh_coordinates_2D[:, 0]
        rho = BlockMesh_coordinates_2D[:, 1]
        theta = np.zeros((len(x),))
        ux = self.mean_fields_dict["ux"]
        ur = self.mean_fields_dict["ur"]
        ut = self.mean_fields_dict["ut"]
        export_path = self.path_to_postprocessing_folder


        distance_to_inlet = np.abs(x-inlet_position)
        indexes_of_inlet = np.where( 
            (distance_to_inlet == np.min(distance_to_inlet)))

        #indexes_of_inlet = np.where( 
        #    (np.abs(x-inlet_position)<4*1e-4))# & 

        rho_inlet = rho[indexes_of_inlet]
        theta_inlet = theta[indexes_of_inlet]
        x_inlet = x[indexes_of_inlet]
        ux_inlet = ux[indexes_of_inlet]
        ur_inlet = ur[indexes_of_inlet]
        ut_inlet = ut[indexes_of_inlet]

        average_axial_velocity = np.trapz(ux_inlet,rho_inlet) / (np.max(rho_inlet)-np.min(rho_inlet))
        self.logger.info(f"inlet average axial velocity (area averaged): {average_axial_velocity:.4f} m/s")

        if self.config["make_plots"]:
            plot_inlet_velocity(inlet_position, x_inlet, rho_inlet, theta_inlet,
                        ur_inlet, ut_inlet, ux_inlet, export_path)

        inlet_conditions = [ux_inlet, ur_inlet, ut_inlet]

        
        return (indexes_of_inlet, inlet_conditions)


    def extract_fuel_injector_velocity(self, BlockMesh_coordinates_2D : np.array, enforce_plotting = True) -> Tuple:
        """
        """
        self.logger.info("now extracting fuel injector position from LES mean field data")

        # parameter definition
        fuel_injector_position_x = float(self.config["fuel_injector_position_x"])
        fuel_injector_position_y = [float(self.config["fuel_injector_position_y_bottom"]), float(self.config["fuel_injector_position_y_top"])]

        x = BlockMesh_coordinates_2D[:, 0]
        rho = BlockMesh_coordinates_2D[:, 1]
        theta = np.zeros((len(x),))
        ux = self.mean_fields_dict["ux"]
        ur = self.mean_fields_dict["ur"]
        ut = self.mean_fields_dict["ut"]
        export_path = self.path_to_postprocessing_folder

        # get the distance to the fuel injector downstream of the duct outlet (the downstream part is ensured by np.where(x<0, -100, x))
        x_distance_to_fuel_injector = np.abs(np.where(x<0, -100, x), x-fuel_injector_position_x)

        indexes_of_fuel_injector_x = np.where((x_distance_to_fuel_injector == np.min(x_distance_to_fuel_injector)))[0]
        indexes_of_fuel_injector_y_1 = np.where((rho <= fuel_injector_position_y[1]))[0]
        indexes_of_fuel_injector_y_2 = np.where((rho >= fuel_injector_position_y[0]))[0]
        indexes_of_fuel_injector_y = np.intersect1d(indexes_of_fuel_injector_y_1, indexes_of_fuel_injector_y_2)
        indexes_of_fuel_injector = np.intersect1d(indexes_of_fuel_injector_x, indexes_of_fuel_injector_y)


        rho_fuel_injector = rho[indexes_of_fuel_injector]
        theta_fuel_injector = theta[indexes_of_fuel_injector]
        x_fuel_injector = x[indexes_of_fuel_injector]
        ux_fuel_injector = ux[indexes_of_fuel_injector]
        ur_fuel_injector = ur[indexes_of_fuel_injector]
        ut_fuel_injector = ut[indexes_of_fuel_injector]

        average_axial_velocity = np.trapz(ux_fuel_injector,rho_fuel_injector) / (np.max(rho_fuel_injector)-np.min(rho_fuel_injector))
        self.logger.info(f"fuel injector average axial velocity (area averaged): {average_axial_velocity:.4f} m/s")

        if self.config["make_plots"] or enforce_plotting:
            plot_inlet_velocity(fuel_injector_position_x, x_fuel_injector, rho_fuel_injector, theta_fuel_injector,
                        ur_fuel_injector, ut_fuel_injector, ux_fuel_injector, export_path, inlet_name="fuel injector")

        fuel_injector_conditions = [ux_fuel_injector, ur_fuel_injector, ut_fuel_injector]

        
        return (indexes_of_fuel_injector, fuel_injector_conditions)
    

    def calculate_production(self, coordinates_2D):

        import matplotlib.pyplot as plt
     
        x_min = -0.1
        x_max = 1
        y_min = 0
        y_max = 0.2
        points_x, points_y = (100,200)
        _x = np.linspace(x_min, x_max, num=points_x)
        _y = np.linspace(y_min, y_max, num=points_y)
        dx, dy = _x[1]-_x[0], _y[1]-_y[0]
        meshgrid_yy, meshgrid_xx = np.meshgrid(_y, _x)
        
        self.gradients = {}
        self.mean_fields_dict_reshaped = {}
        self.mean_fields_dict
        for key in self.mean_fields_dict.keys():
            plt.close()
            plt.figure()
            plt.imshow(self.mean_fields_dict[key].reshape((200,100)))
            fig.tight_layout()
            plt.savefig(f"{self.path_to_postprocessing_folder}//mean_field_{key}_FIELD.png", dpi=300, bbox_inches='tight')

        divnorm=colors.TwoSlopeNorm(vmin=-5., vcenter=0., vmax=10)
        # pcolormesh(your_data, cmap="coolwarm", norm=colors.TwoSlopeNorm(vmin=-5., vcenter=0., vmax=10))

        # DUMMY
        dummy = 0.2 * meshgrid_xx - 0.3 * meshgrid_yy 
        dummy_grad = np.gradient(dummy, dx, dy)
        key="dummy"
        plt.close()
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9,9))
        sc=axes[0].imshow(dummy, origin='lower',aspect=0.2/1.1, cmap="coolwarm")
        axes[0].set_title(f"field: {key}")
        sc2=axes[1].imshow(dummy_grad[0], origin='lower',aspect=0.2/1.1, cmap="coolwarm")
        axes[1].set_title(f"gradient in x")
        sc3=axes[2].imshow(dummy_grad[1], origin='lower',aspect=0.2/1.1, cmap="coolwarm")
        axes[2].set_title(f"gradient in r")
        fig.colorbar(sc, ax=axes[0])
        fig.colorbar(sc2, ax=axes[1])
        fig.colorbar(sc3, ax=axes[2])
        fig.tight_layout()
        plt.savefig(f"{self.path_to_postprocessing_folder}//mean_field_and_gradients_{key}.png", dpi=300, bbox_inches='tight')
        print(f"done with {key}")

        for key in self.mean_fields_dict.keys():
            print(f"now doing {key}")
            field = self.mean_fields_dict[key].reshape((200,100))
            self.mean_fields_dict_reshaped[key] = field
            # gradient = np.gradient(field, delta_x, delta_r)
            # gradient = np.gradient(field, coordinates_2D)
            gradient = np.gradient(field, dx, dy)
            self.gradients[f"d{key}_dx"] = gradient[0]
            self.gradients[f"d{key}_dr"] = gradient[1]

            # plt.close()
            # plt.figure()
            # plt.imshow(gradient[0], origin='lower')
            # fig.tight_layout()
            # plt.savefig(f"{self.path_to_postprocessing_folder}//grad_{key}_FIELD.png", dpi=300, bbox_inches='tight')
            if np.min(field) == 0.0:
                divnorm_data = colors.LogNorm(vmin=0, vmax=np.max(field))
            else:
                divnorm_data = colors.TwoSlopeNorm(vmin=np.min(field), vcenter=0., vmax=np.max(field))
            # divnorm = colors.TwoSlopeNorm(vmin=np.min(gradient), vcenter=0., vmax=np.max(gradient))
            divnorm = colors.TwoSlopeNorm(vmin=-10, vcenter=0., vmax=10)

            plt.close()
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9,9))
            sc=axes[0].imshow(field, origin='lower',aspect=0.2/1.1, cmap="coolwarm")#, norm=divnorm_data)
            axes[0].set_title(f"field: {key}")
            sc2=axes[1].imshow(gradient[0], origin='lower',aspect=0.2/1.1, cmap="coolwarm", norm=divnorm)
            axes[1].set_title(f"gradient in x")
            sc3=axes[2].imshow(gradient[1], origin='lower',aspect=0.2/1.1, cmap="coolwarm", norm=divnorm)
            axes[2].set_title(f"gradient in r")
            fig.colorbar(sc, ax=axes[0])
            fig.colorbar(sc2, ax=axes[1])
            fig.colorbar(sc3, ax=axes[2])
            fig.tight_layout()
            plt.savefig(f"{self.path_to_postprocessing_folder}//mean_field_and_gradients_{key}.png", dpi=300, bbox_inches='tight')
            print(f"done with {key}")
            

            # plt.close()
            # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9,3), )
            # sc=axes[0].scatter(coordinates_2D[:,0], coordinates_2D[:,1], s=2, c=self.mean_fields_dict[key], alpha=1.0)
            # axes[0].set_title(f"field: {key}")
            # sc2=axes[1].scatter(coordinates_2D[:,0], coordinates_2D[:,1], s=2, c=self.gradients[f"d{key}_dx"], alpha=1.0)
            # axes[1].set_title(f"gradient in x")
            # sc3=axes[2].scatter(coordinates_2D[:,0], coordinates_2D[:,1], s=2, c=self.gradients[f"d{key}_dr"], alpha=1.0)
            # axes[2].set_title(f"gradient in r")
            # fig.colorbar(sc, ax=axes[0])
            # fig.colorbar(sc2, ax=axes[1])
            # fig.colorbar(sc3, ax=axes[2])
            # fig.tight_layout()
            # plt.savefig(f"{self.path_to_postprocessing_folder}//mean_field_and_gradients_{key}.png", dpi=300, bbox_inches='tight')
            # print(f"done with {key}")
        
        P = np.zeros((100,100))
        self.turbulence_production = {}
        text = ""
        for i in ["x", "r"]:
            for j in ["x", "r"]:
                if f"u{i}{j}" in self.mean_fields_dict.keys():
                    text_for_this_component = f"- <u{i}u{j}> * dU{i} / d{j}"
                    text += text_for_this_component + "\n"
                    P_comp = np.multiply(-self.mean_fields_dict_reshaped[f"u{i}{j}"],self.gradients[f"du{i}_d{j}"])
                    self.turbulence_production[f"{i}{j}"] = P_comp
                    P = np.add(P, P_comp)

                    plt.close()
                    fig = plt.figure(figsize=(4,4))
                    sc = plt.scatter(coordinates_2D[:,0], coordinates_2D[:,1], s=2, c=P_comp, alpha=1.0)
                    fig.colorbar(sc)
                    plt.title(f"- <u{i}u{j}> * dU{i} / d{j}")
                    plt.savefig(f"{self.path_to_postprocessing_folder}//turbulence_production_{i}{j}.png", dpi=300, bbox_inches='tight')

        self.turbulence_production_total = P
        plt.close()
        fig = plt.figure(figsize=(4,4))
        sc = plt.scatter(coordinates_2D[:,0], coordinates_2D[:,1], s=2.5, c=P, alpha=0.8)
        fig.colorbar(sc)
        plt.title(f"- <uiuj> * dUi / dj")
        plt.savefig(f"{self.path_to_postprocessing_folder}//turbulence_production_total.png", dpi=300, bbox_inches='tight')


    def calculate_production_old(self):
        """
        this function calculates the turblence production
        """
        from scipy.interpolate import griddata
        import matplotlib.pyplot as plt
    


        # Load the mesh using meshio
        # mesh = meshio.read(self.config["path_to_LES_results"])

        gradients = {}

        coordinates_orig = self.coordinates
        self.logger.info(f"coordinates: {coordinates_orig.shape}")
        
        x_min = -0.1
        x_max = 1
        y_min = -0.2
        y_max = 0.2
        z_min = y_min
        z_max = y_max

        self.logger.info("calculating turbulence production")

        point_inside_x = np.logical_and((coordinates_orig[:,0] >= x_min), (coordinates_orig[:,0] <= x_max))
        point_inside_y = np.logical_and((coordinates_orig[:,1] >= y_min), (coordinates_orig[:,1] <= y_max))
        point_inside_z = np.logical_and((coordinates_orig[:,2] >= z_min), (coordinates_orig[:,2] <= z_max))

        point_in_cell_batch = np.logical_and(point_inside_x, point_inside_y)
        point_in_cell_batch = np.logical_and(point_in_cell_batch, point_inside_z)
        coordinates =coordinates_orig[point_in_cell_batch]

        points_x, points_y, points_z = (20,20,20)
        _x = np.linspace(x_min, x_max, num=points_x)
        _y = np.linspace(y_min, y_max, num=points_y)
        _z = np.linspace(z_min, z_max, num=points_z)
        meshgrid_xx, meshgrid_yy, meshgrid_zz = np.meshgrid(_x, _y, _z)


        print(f"points reduced to {np.sum(point_in_cell_batch)}")
        interpolation_dict = {}
        for key in self.velocities.keys():
            print(f"now doing {key}")
        
            data_orig = self.velocities[key]
            data = data_orig[point_in_cell_batch]

            interpolation = griddata(coordinates, data, (meshgrid_xx, meshgrid_yy, meshgrid_zz), method="nearest")  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
            interpolation_dict[key] = interpolation
            gradients[f"d{key}_dx"] = np.gradient(interpolation, _x[1]-_x[0], axis=0)
            gradients[f"d{key}_dy"] = np.gradient(interpolation, _y[1]-_y[0], axis=1)
            gradients[f"d{key}_dz"] = np.gradient(interpolation, _z[1]-_z[0], axis=2)
            print(f"done with {key}")

        print(interpolation.shape)

        P = np.zeros(interpolation.shape)
        text = ""
        for i in ["x", "y", "z"]:
            for j in ["x", "y", "z"]:
                if f"u{i}{j}" in interpolation_dict.keys():
                    text += f"- <u{i}u{j}> * dU{i} / d{j}\n"
                    P += -interpolation_dict[f"u{i}{j}"]*gradients[f"du{i}_d{j}"]

        print(text)

    def calculate_mean_field_dissipation(self):

        self.logger.info("calculating mean_field_dissipation")
        self.logger.info(f"read mesh: {self.path_to_LES_results}")
        mesh = meshio.read(self.path_to_LES_results)
        self.mesh = mesh
        self.logger.info(f"finished importing mesh")
        self.mean_field_dissipation_simple = - self.mean_fields_dict["uxr"] * self.mean_fields_dict["delUx_delr"]
        self.mean_field_dissipation_full = - self.mean_fields_dict["uxr"] * 0.5 * (self.mean_fields_dict["delUx_delr"] + self.mean_fields_dict["delUr_delx"])
        self.eddy_viscosity_from_inversion_simple = - self.mean_fields_dict["uxr"] / self.mean_fields_dict["delUx_delr"]
        self.eddy_viscosity_from_inversion_full = - self.mean_fields_dict["uxr"] /( 0.5 * (self.mean_fields_dict["delUx_delr"] + self.mean_fields_dict["delUr_delx"]))
        
        
        direction_names = [f"{i}{j}" for i in ["x","y","z"] for j in ["x","y","z"]]
        self.U_mean = dict(zip(["x","y","z"], np.array(mesh.point_data['UMean']).T))
        # the direction names can be found here: https://discourse.paraview.org/t/definition-of-gradients/4767 
        self.U_mean_grad = dict(zip(direction_names, np.array(mesh.point_data['UMean_Gradient']).T))
        self.uu_dash_mean = dict(zip(["xx","yy","zz", "xy", "yz", "xz"], np.array(mesh.point_data['UPrime2Mean']).T))
        self.mean_field_dissipation = {}
        self.mean_rate_of_strain = {}
        self.eddy_viscosity_from_inversion_component = {}
        for ij in ["xx","yy","zz", "xy", "yz", "xz"]:
            i, j = tuple(ij) 
            self.mean_rate_of_strain[f"{ij}"] = 0.5 * (self.U_mean_grad[f"{ij}"] + self.U_mean_grad[f"{j}{i}"])
            self.mean_field_dissipation[f"{ij}"] = -self.uu_dash_mean[f"{ij}"]*self.mean_rate_of_strain[f"{ij}"]
            self.eddy_viscosity_from_inversion_component[f"{ij}"] = - self.uu_dash_mean[f"{ij}"] / self.mean_rate_of_strain[f"{ij}"]

        self.mean_field_dissipation_total_cartesian = np.sum(np.array(list(self.mean_field_dissipation.values())), axis=0)
        self.eddy_viscosity_from_inversion_full_cartesian = np.sum(np.array(list(self.eddy_viscosity_from_inversion_component.values())), axis=0)
        self.eddy_viscosity_from_inversion_simple_cartesian = - self.uu_dash_mean["xy"] / self.U_mean_grad["xy"]

        # # slice
        # self.coordinates = np.array(mesh.points)
        # ###########################################
        # z_margin = 0.001
        # x_min = -0.1
        # x_max = 1
        # y_min = 0
        # y_max = 0.2
        # coordinate_filter_for_close_up_limits = np.where((self.coordinates[:,0] >= x_min)\
        #                                     & (self.coordinates[:,0] <= x_max)\
        #                                     & (self.coordinates[:,1] >= y_min)\
        #                                     & (self.coordinates[:,1] <= y_max)\
        #                                     & (self.coordinates[:,2] >= -z_margin)\
        #                                     & (self.coordinates[:,2] <= z_margin))[0]
        # coordinates_sliced = self.coordinates[coordinate_filter_for_close_up_limits]
        # self.coordinates_sliced = coordinates_sliced
        # self.coordinate_filter_for_center_slice = coordinate_filter_for_close_up_limits
        # c = self.turbulence_production_total[coordinate_filter_for_close_up_limits]
        # self.turbulence_production_sliced = c
        # divnorm=colors.TwoSlopeNorm(vmin=np.min(c), vcenter=0., vmax=np.max(c))
        # plt.close()
        # plt.figure(figsize=(9,4))
        # x = coordinates_sliced[:,0]
        # y = coordinates_sliced[:,1]
        # sc=plt.scatter(x, y, s=1, c=c, alpha=0.8, cmap="coolwarm", norm=divnorm)
        # plt.colorbar(sc)
        # plt.title("turbulence production")
        # plt.savefig(f"{self.path_to_postprocessing_folder}/TurbProd_sliced.png", dpi=300)

        # for ij in ["xx","yy","zz", "xy", "yz", "xz"]:
        #     i = ij[0]
        #     j = ij[1]
        #     plt.close()
        #     fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9,9))
        #     x = coordinates_sliced[:,0]
        #     y = coordinates_sliced[:,1]
        #     c1 = self.uu_dash_mean[f"{ij}"][coordinate_filter_for_close_up_limits]
        #     c2 = self.U_mean_grad[f"{ij}"][coordinate_filter_for_close_up_limits]
        #     c3 = self.turbulence_production[f"{ij}"][coordinate_filter_for_close_up_limits]
        #     sc1=axes[0].scatter(x, y, s=1, c=c1, alpha=0.5)
        #     sc2=axes[1].scatter(x, y, s=1, c=c2, alpha=0.5)
        #     sc3=axes[2].scatter(x, y, s=1, c=c3, alpha=0.5)
        #     axes[0].set_title(fr"$<u_{i}^\prime u_{j}^\prime >$")
        #     axes[1].set_title(fr"$U_{i} / d{j}$")
        #     axes[2].set_title("production")
        #     fig.colorbar(sc1, ax=axes[0])
        #     fig.colorbar(sc2, ax=axes[1])
        #     fig.colorbar(sc3, ax=axes[2])
        #     fig.suptitle(f"turbulence production (slice through domain) in {ij}")
        #     fig.tight_layout()  
        #     plt.savefig(f"{self.path_to_postprocessing_folder}turb_production_{ij}_sliced.png")





