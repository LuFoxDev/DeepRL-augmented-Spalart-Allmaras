from click import progressbar
from tensorboard import program
import pyvista as pv
import numpy as np
from scipy import integrate

pv.global_theme.return_cpos = True

path_to_les_results = "/home/lukas/tubCloud/Documents/Ubuntu_shared/Jetflow/meanFields_Re15k_Cold.vtk"
#sphere = pv.Sphere(center=(-0.4, -0.4, -0.4))

mesh = pv.read(path_to_les_results)
cpos = [(0, 0, 0.5), (0, 0, -2), (0, 1, 0)]
#mesh_half = mesh.clip(normal=(0,0,-1), crinkle=True, invert=False)
#newcpos = mesh.plot(cpos=cpos, show_edges=True, color=True, background="white", anti_aliasing=True, screenshot=True, return_cpos=True)
#print(newcpos)
#mesh_half.plot(cpos=cpos, show_edges=True, color=True, background="white", anti_aliasing=True, screenshot=True)


# labels = dict(zlabel='z', xlabel='x', ylabel='y', color="black")
# p = pv.Plotter(window_size=[1280,1280])
# mesh_half = mesh.clip(normal=(0,0,-1), crinkle=True, invert=False)
# p.add_mesh(mesh_half, color="grey", show_edges=True)
# p.background_color = 'w'
# p.enable_anti_aliasing()
# p.show_grid(**labels)
# p.add_axes(**labels)
# p.set_position((-2, 2, 2))
# #p.remove_scalar_bar()
# #p.add_camera_orientation_widget()
# p.show()


# compute volumes
mesh = pv.read(path_to_les_results)
#cell_sizes = mesh.compute_cell_sizes(length=False, area=True, volume=True)
#volumes = cell_sizes.get_array("Volume")
#areas = cell_sizes.get_array("Area")


U = np.array(mesh.point_data['UMean'])
ux = U[:, 0]
uy = U[:, 1]
uz = U[:, 2]
coordinates = mesh.points
coo_LES = np.array(mesh.points)
x = coo_LES[:, 0]
y = coo_LES[:, 1]
z = coo_LES[:, 2]

############################################
# main inlet bulk velocity
inlet_position = np.min(x)
distance_to_inlet = np.abs(x-inlet_position)
indexes_of_inlet = np.where( 
    (distance_to_inlet == np.min(distance_to_inlet)))

y_inlet = y[indexes_of_inlet]
z_inlet = z[indexes_of_inlet]
ux_inlet = ux[indexes_of_inlet]
#slice through y
bulk_velocity = 0
for _y_coor in np.unique(y_inlet):
    distance_to_slice = np.abs(y_inlet-_y_coor)
    indexes_of_slice = np.where( 
        (distance_to_slice == np.min(distance_to_slice)))
    if len(indexes_of_slice[0]) > 1:
        ux_slice = ux_inlet[indexes_of_slice]
        z_slice = z_inlet[indexes_of_slice]
        area = np.trapz(ux_slice,z_slice)
        bulk_velocity += area

mean_velocity = np.mean(ux[indexes_of_inlet])
print(f"mean_velocity: {mean_velocity:.3f} m/s") 
print(f"bulk_velocity: {bulk_velocity:.3f} m/s") 

