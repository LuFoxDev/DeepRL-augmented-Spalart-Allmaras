import numpy as np
#import scipy.io
#import h5py
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

import os
import glob
import meshio
#import matplotlib.cm as cm
#import matplotlib.tri as tri
#from scipy import interpolate
import pickle

import modules.FF_funs as ff
from modules.extract_inlet_velocity import *
from modules.read_OF_coordinates import *
from modules.plotting_functions import *

path_to_LES_results = "/home/lukas/tubCloud/Documents/Ubuntu_shared/Jetflow/meanFields_Re10k_Cold.vtk"
path_for_image_exports = "./exports/images/"
path_to_OF_geometry_data = "/home/lukas/Data/OF-Tests/simpleFoamCase/templates/"
path_to_simulation_results = "/home/lukas/Data/OF-Tests/simpleFoamCase/6000/"

#path_to_blockmesh_atT0 = "/home/lukas/Data/OF-Tests/simpleFoamCase/VTK/simpleFoamCase_0.vtk"

#######################################################################

if not os.path.exists(path_for_image_exports):
    full_image_path = os.path.join(os.getcwd(), path_for_image_exports)
    if "/./" in full_image_path:
        full_image_path = full_image_path.replace("/./", "/")
    os.makedirs(full_image_path)

inlet_position = -0.12

# plot velocity fields
close_up_limits = {
    "x" : (-0.1, 0.1),
    "y" : (0.015, 0.02)
}

write_inlet_boundary_condition = False

#######################################################################
# Import blockmesh data
print("read blockmesh coordinates")
BlockMesh_coordinates = read_OF_internal_field_coordinates(path_to_OF_geometry_data)
BlockMesh_coordinates_2D = BlockMesh_coordinates[:,[0,1]]

plot_coordinates(BlockMesh_coordinates[:, 0], BlockMesh_coordinates[:, 1],
                 BlockMesh_coordinates[:, 2], source="own mesh_non vtk",
                 export_path=path_for_image_exports, show_all=True)


#######################################################################
# Import vtk LES jetflow data
print(f"read mesh: {path_to_LES_results}")
mesh = meshio.read(path_to_LES_results)

U = np.array(mesh.point_data['UMean'])
ux = U[:, 0]
uy = U[:, 1]
uz = U[:, 2]
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

plot_coordinates(x, y, z, source="LES data",
                 export_path=path_for_image_exports)

# coordinate transform
Z = y + 1j * z
def z2polar(z): return (abs(z), np.angle(z))

rho, theta = z2polar(Z)
# define polar coordinates x, rho, theta
coo_old = np.zeros((len(rho), 3))
coo_old[:, 0] = x
coo_old[:, 1] = rho
coo_old[:, 2] = theta

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

#############################################################################
# extend in azimuthal dimension
print("coo_blockMesh_2D_exp_car = ff.ExpandForAzimuthalAverage(coo_blockMesh_2D,100)")
coo_blockMesh_2D_exp_car = ff.ExpandForAzimuthalAverage(BlockMesh_coordinates_2D, 100)
plot_coordinates(coo_blockMesh_2D_exp_car[:, 0], coo_blockMesh_2D_exp_car[:, 1],
                 coo_blockMesh_2D_exp_car[:, 2], source="coo_blockMesh_2D_exp_car", export_path=path_for_image_exports)
# this is still in x y z

coo_blockMesh_2D_exp = np.zeros((len(coo_blockMesh_2D_exp_car[:, 1]), 3))
# coor trafo
Z = coo_blockMesh_2D_exp_car[:, 1] + 1j * coo_blockMesh_2D_exp_car[:, 2]
rho_exp, theta_exp = z2polar(Z)
coo_blockMesh_2D_exp[:, 0] = coo_blockMesh_2D_exp_car[:, 0]
coo_blockMesh_2D_exp[:, 1] = rho_exp
coo_blockMesh_2D_exp[:, 2] = theta_exp

# stack variables for interpolation
temp = np.zeros((len(ux), 9))
temp[:, 0] = ux
temp[:, 1] = ur
temp[:, 2] = ut
temp[:, 3] = uxx
temp[:, 4] = urr
temp[:, 5] = utt
temp[:, 6] = uxr
temp[:, 7] = uxt
temp[:, 8] = urt

# interpolate
# first linear
#print("now interpolating (linear)")
#temp_exp = np.squeeze(interpolate.griddata(coo_LES, temp, coo_blockMesh_2D_exp_car, method='linear'))
# second nearest for the nan points
print("now interpolating (nearest)")
temp_exp = np.squeeze(interpolate.griddata(
    coo_LES, temp, coo_blockMesh_2D_exp_car, method='nearest'))
# temp_exp[np.isnan(temp_exp)]=temp_exp2[np.isnan(temp_exp)]

# azimuthal averaging (this is again on cartesian coordinates)
print("azimuthal averaging")
mean_field_2D = np.zeros((len(BlockMesh_coordinates_2D[:,0]), 9))
for ind in range(9):
    mean_field_2D[:, ind] = ff.ContractFromAximuthalAverage(
        BlockMesh_coordinates_2D, 100, temp_exp[:, ind])

ux_2D = mean_field_2D[:, 0]
ur_2D = mean_field_2D[:, 1]
ut_2D = mean_field_2D[:, 2]
uxx_2D = mean_field_2D[:, 3]
urr_2D = mean_field_2D[:, 4]
utt_2D = mean_field_2D[:, 5]
uxr_2D = mean_field_2D[:, 6]
uxt_2D = mean_field_2D[:, 7]
urt_2D = mean_field_2D[:, 8]

mean_fields = {'ux': mean_field_2D[:, 0], 'ur': mean_field_2D[:, 1], 'ut': mean_field_2D[:, 2], 'uxx': mean_field_2D[:, 3],
               'urr': mean_field_2D[:, 4], 'utt': mean_field_2D[:, 5], 'uxr': mean_field_2D[:, 6], 'uxt': mean_field_2D[:, 7], 'urt': mean_field_2D[:, 8]}

print("now dumping mean fields")
pickle.dump(mean_fields, open("exports/meanfields.p", "wb"))

# --------------------------------------------------
# plot velocity fields o mean field
for i, n in zip(range(3), ["Ux", "Uy", "Uz"]):
    c = mean_field_2D[:,i].reshape(BlockMesh_coordinates_2D[:,0].shape)
    plot_scalar_field_on_mesh(BlockMesh_coordinates_2D[:,0], BlockMesh_coordinates_2D[:,1], c, f"mean field (LES data): {n}",
                              "x", "y", f"LES_mean_field_{n}", path_for_image_exports, close_up_limits)

#######################################################################
# Extract inlet velocities from LES data
# --------------------------------------------------------------

if write_inlet_boundary_condition:
    x = BlockMesh_coordinates_2D[:, 0]
    y = BlockMesh_coordinates_2D[:, 1]
    z = np.zeros((len(x),))

    # get inlet position from mean field data from LES mean field
    indexes_of_inlet = extract_inlet_velocity(
        inlet_position, x, y, z, ux_2D, ur_2D, ut_2D, export_path=path_for_image_exports)

    # get inlet position from open foam mesh
    sim_inlet_coordinates = read_OF_inlet_coordinates(path_to_OF_geometry_data)

    # interpolate from mean field data to open foam mesh coordinates for inlet condition
    inlet_conditions = interpolate_inlet_condition_data(x, y, ux_2D, ur_2D, ut_2D, indexes_of_inlet, sim_inlet_coordinates, export_path=path_for_image_exports)

    # write inlet velocity data to open foam files
    write_OF_boundary_condition(sim_inlet_coordinates, inlet_conditions)
    #######################################################################


#######################################################################
# Import results
simulation_results = read_OF_internal_field_results("U", path_to_simulation_results)

for i, n in zip(range(3), ["Ux", "Uy", "Uz"]):
    c = simulation_results[:,i].reshape(BlockMesh_coordinates_2D[:,0].shape)
    plot_scalar_field_on_mesh(BlockMesh_coordinates_2D[:,0], BlockMesh_coordinates_2D[:,1], c, f"mean field (LES data): {n}",
                              "x", "y", f"simulation_results_{n}", path_for_image_exports, close_up_limits)

#############################################################################
# -------------------------------------------------------------
# Calulating error
print("Calulating error")
rmse_by_time = []
path_to_OF_case = '/home/lukas/Data/OF-Tests/simpleFoamCase/'
all_elements_in_OF_folder = glob.glob(path_to_OF_case + "*")
results_folder = [folder for folder in all_elements_in_OF_folder if os.path.isfile(folder + "/U") and folder[-2:] != '/0']
time_list = [int(f.split("/")[-1]) for f in results_folder]
time_list.sort()

for time in time_list:
    path_to_simulation_results_by_time = f'{path_to_OF_case}{time}/'
    simulation_results = read_OF_internal_field_results("U", path_to_simulation_results_by_time)
    absolute_error_by_dir = simulation_results[:,0:2] - mean_field_2D[:,0:2]  # (31500, 2)
    error = np.sqrt(np.sum(absolute_error_by_dir, axis=1)**2)
    rmse = np.sum(error)/error.shape[0]
    rmse_by_time.append(rmse)

import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(time_list, rmse_by_time)
plt.xlabel("epoch")
plt.ylabel("error")
plt.title("rmse over epoch")
fig.savefig(path_for_image_exports+"/rmse_by_time.png", dpi=300)

for i, error_parameter in zip(range(3), ["Ux", "Uy", "Uz"]):
    LES_mean_field_data_for_error = mean_field_2D[:,i].reshape(BlockMesh_coordinates_2D[:,0].shape)
    simulation_data_for_error = simulation_results[:,i].reshape(BlockMesh_coordinates_2D[:,0].shape)

    # RMS
    # squared error
    absolute_error = simulation_data_for_error - LES_mean_field_data_for_error
    error = np.sqrt(absolute_error**2)
    rmse = np.sum(error)/error.shape[0]
    plot_scalar_field_on_mesh(BlockMesh_coordinates_2D[:,0], BlockMesh_coordinates_2D[:,1],
                                absolute_error,
                                f"absolute error based on {error_parameter} (rmse = {rmse:.3f})\npositive values mean SA predicts higher values than LES mean field",
                                "x", "y", f"absolute_error_{error_parameter}",
                                path_for_image_exports,
                                close_up_limits, cmap="seismic", center_cbar=True)

    print("mean squared error: {rmse}")
    plot_scalar_field_on_mesh(BlockMesh_coordinates_2D[:,0], BlockMesh_coordinates_2D[:,1], 
                            error,f"error based on {error_parameter} (rmse = {rmse:.3f})",
                            "x", "y", f"error_{error_parameter}",
                            path_for_image_exports,
                            close_up_limits)
