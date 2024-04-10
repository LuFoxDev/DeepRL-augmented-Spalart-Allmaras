from xmlrpc.client import Boolean
import matplotlib.pyplot as plt
import matplotlib as mlp
import matplotlib.tri as tri
import matplotlib as mlp
import matplotlib.colors as colors
import matplotlib.tri as tr
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from scipy.interpolate import make_interp_spline, BSpline

import numpy as np
from scipy.integrate import simps
from modules.utilities.logging import logging
import os
from typing import Union

logger = logging.getLogger("openfoam_case")

# mlp.use("pgf") # mlp.use("agg")
# mlp.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
#     'mathtext.fontset' : 'dejavuserif'
# })

mlp.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset' : 'dejavuserif'
})

def plot_scalar_field_on_mesh(x, y, c, title, x_label, y_label, c_label, filename, filepath, close_up_limits=None, close_up_aspect_ratio="equal", cmap="RdBu_r", center_cbar=False, aspect_ratio=None, custom_axis_limits=None, levels_overwrite=None, manual_clim=None, flip_y_axis=False):
    """
    this function creates a triangulation of the mesh coordinates
    then the function ymax(x) = max(y(x)) is calculated
    this function is used to mask out unwanted triangles that appear at the edge of the mesh
    next create a tricontourf plot
    """

    # sanity check
    data_shapes_ok = True
    for _data in [x, y, c]:
        if len(_data.shape) == 0:
            data_shapes_ok = False
            logger.warning(f"cannot plot {title} because shapes do not match: x: {x.shape}, y: {y.shape}, c: {c.shape}")

    if data_shapes_ok:
        if (x.shape[0] == y.shape[0]) and (x.shape[0] == c.shape[0]):

            if len(x.shape) != 1:
                x = x.reshape((x.shape[0],))
                logger.info(f"reshaped x data for plotting {title}")
            if len(y.shape) != 1:
                y = y.reshape((x.shape[0],))
                logger.info(f"reshaped y data for plotting {title}")
            if len(c.shape) != 1:
                c = c.reshape((x.shape[0],))
                logger.info(f"reshaped c data for plotting {title}")
            data_shapes_ok = True

        else:
            data_shapes_ok = False
            logger.warning(f"cannot plot {title} because shapes do not match: x: {x.shape}, y: {y.shape}, c: {c.shape}")

    if data_shapes_ok:
        # #-----------------------------------------
        # if aspect_ratio=="equal":
        #     aspect_ratio = 1

        # if aspect_ratio=="auto":
        #     aspect_ratio = 2

        #-----------------------------------------
        # perform triangulation
        triang = tri.Triangulation(x, y)
        triangles = triang.triangles

        #-----------------------------------------
        # extract the maximum y values per x coordinate
        max_y_by_x = {}
        unique_x = np.unique(np.array(x))
        for x_value in unique_x:
            max_y_by_x[x_value] = np.max(y[np.where(x==x_value)])
        
        #-----------------------------------------
        # masking
        mask = []
        for points in triangles:
            #print points
            triang_is_not_inside = False
            #x_mean = np.mean(x[points])
            y_mean = np.mean(y[points])
            triang_is_not_inside = np.min([max_y_by_x[x_] for x_ in x[points]]) < y_mean
            mask.append(triang_is_not_inside)
        
        triang.set_mask(mask)
        
        #-----------------------------------------
        # Tricontour plot
        # Directly supply the unordered, irregularly spaced coordinates to tricontour.
        width = 8
        if aspect_ratio is None:
            used_aspect_ratio = (np.max(x)-np.min(x))/(np.max(y)-np.min(y))
        else: 
            used_aspect_ratio = aspect_ratio
        heigth =  width / used_aspect_ratio
        plt.close()
        fig, ax = plt.subplots(figsize=(width, heigth))

        # plot all triangles
        # ax.triplot(triang, color='red', lw=0.01)
        # ax.tricontour(x, y, z, levels=100, linewidths=0.1, colors='k')
        if center_cbar:
            max_abs = np.round(np.max(np.abs(c)), 1)
            vmin = -max_abs
            vmax = max_abs
            levels = np.linspace(vmin, vmax, 100)
            ticks_list = []
            ticks_list.append(vmin)
            ticks_list = [vmin] + list(np.arange(np.ceil(vmin)+1, np.floor(vmax),1)) + [vmax] 
        elif manual_clim is not None:
            
            if len(manual_clim)>2:
                levels = np.linspace(np.min(manual_clim), np.max(manual_clim), 100)
                # ticks_list = [vmin] + list(np.arange(np.ceil(vmin)+1, np.floor(vmax),tick_size)) + [vmax]
                ticks_list = manual_clim
            else:
                    
                cbar_range =  manual_clim[1]-manual_clim[0]
                # tick_size = np.round(cbar_range/5, decimals=1)
                decimals = int(np.ceil(np.abs(np.log10(cbar_range))))
                if (manual_clim[0] >= 0) and (manual_clim[0] < 1e-5):
                    vmin = 0  #np.np.floor(manual_clim[0])
                else:
                    vmin = np.round(manual_clim[0], decimals=decimals) - 1*10**(-decimals)  #np.np.floor(manual_clim[0])
                if (manual_clim[1] >= 0) and (manual_clim[1] < 1e-5):
                    vmax = 0
                else:
                    vmax = np.round(manual_clim[1], decimals=decimals) + 1*10**(-decimals)  #np.ceil(manual_clim[1])
                levels = np.linspace(vmin, vmax, 100)
                # ticks_list = [vmin] + list(np.arange(np.ceil(vmin)+1, np.floor(vmax),tick_size)) + [vmax]
                ticks_list = np.linspace(start=vmin, stop=vmax, num=5)
        else:
            vmin = np.min(c)
            vmax = np.max(c)
            levels = 200

        if levels_overwrite is not None:
            levels=levels_overwrite

        cntr = ax.tricontourf(triang, c, levels=levels, cmap=cmap)#, norm=colors.Normalize(vmin=vmin, vmax=vmax))
        if aspect_ratio is not None:
            ax.set_aspect(aspect_ratio)
        else:
            ax.set_aspect('equal')
        #cntr.set_clim(-5,5)

        #ax.set_aspect(aspect_ratio)

        axins = inset_axes(ax, width = "5%", height = "100%", loc = 'lower left',
                   bbox_to_anchor = (1.02, 0., 1, 1), bbox_transform = ax.transAxes,
                   borderpad = 0)

        if center_cbar:
            cbar = fig.colorbar(cntr, cax=axins, ticks=ticks_list)#, shrink=1/aspect_ratio, pad=0.05)
        elif manual_clim is not None: 
            cbar = fig.colorbar(cntr, cax=axins, ticks=ticks_list)#, shrink=1/aspect_ratio, pad=0.05)
        else:
            cbar = fig.colorbar(cntr, cax=axins)#, shrink=1/aspect_ratio, pad=0.05)

        #caxis.min = vmin
        #ax2.plot(x, y, 'ko', ms=1)
        #ax2.set(xlim=(-2, 2), ylim=(-2, 2))
        title_mathmode = title.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
        ax.set_title(title_mathmode)
        x_label_mathmode = x_label.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
        ax.set_xlabel(x_label_mathmode)
        y_label_mathmode = y_label.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
        ax.set_ylabel(y_label_mathmode)
        c_label_mathmode = c_label.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
        cbar.ax.set_ylabel(c_label_mathmode)

        if custom_axis_limits is not None:
            ax.set_ylim(custom_axis_limits['y'])
            ax.set_xlim(custom_axis_limits['x'])
            #plt.tight_layout()

        if flip_y_axis:
            ax.set_ylim(ax.get_ylim()[::-1])
            
        save_figure_in_multiple_formats(plt.gcf(), f"{filepath}{filename}", (5,6))
        # plt.savefig(f"{filepath}{filename}.png", dpi=300, bbox_inches='tight')
        # fig.set_figheight(5)
        # fig.set_figwidth(6)
        # plt.savefig(f"{filepath}{filename}.pdf", dpi=300, bbox_inches='tight')
        # plt.savefig(f"{filepath}{filename}.pgf", bbox_inches="tight")
        #plt.savefig(f"{filepath}{filename}.svg", bbox_inches='tight')

        logger.info("saved: " + f"{filepath}{filename}.png")

        if close_up_limits is not None:
            ax.set_ylim(close_up_limits['y'])
            ax.set_xlim(close_up_limits['x'])
            ax.set_aspect(close_up_aspect_ratio)
            if flip_y_axis:
                ax.set_ylim(ax.get_ylim()[::-1])

            save_figure_in_multiple_formats(plt.gcf(), f"{filepath}{filename}_closeup", (5,6))
            # plt.savefig(f"{filepath}{filename}_closeup.png", dpi=300, bbox_inches='tight')
            # plt.savefig(f"{filepath}{filename}_closeup.pdf", dpi=300, bbox_inches='tight')
            # fig.set_figheight(5)
            # fig.set_figwidth(6)
            # plt.savefig(f"{filepath}{filename}_closeup.pgf", bbox_inches="tight")
            #plt.savefig(f"{filepath}{filename}_closeup.svg", bbox_inches='tight')

            logger.info("saved: " + f"{filepath}{filename}_closeup.png")

        
        plt.close()


def plot_scalar_field_with_streamlines_on_mesh(x, y, ux, uy, c, title, x_label, y_label, c_label, filename, filepath, close_up_limits=None, close_up_aspect_ratio="equal", cmap="RdBu_r", center_cbar=False, aspect_ratio=None, custom_axis_limits=None, levels_overwrite=None, manual_clim=None, flip_y_axis=False, streamline_color="white"):
    """
    this function creates a triangulation of the mesh coordinates
    then the function ymax(x) = max(y(x)) is calculated
    this function is used to mask out unwanted triangles that appear at the edge of the mesh
    next create a tricontourf plot
    """
    #mlp.use("pgf") 
    # sanity check
    data_shapes_ok = True
    for _data in [x, y, c]:
        if len(_data.shape) == 0:
            data_shapes_ok = False
            logger.warning(f"cannot plot {title} because shapes do not match: x: {x.shape}, y: {y.shape}, c: {c.shape}")

    if data_shapes_ok:
        if (x.shape[0] == y.shape[0]) and (x.shape[0] == c.shape[0]):

            if len(x.shape) != 1:
                x = x.reshape((x.shape[0],))
                logger.info(f"reshaped x data for plotting {title}")
            if len(y.shape) != 1:
                y = y.reshape((x.shape[0],))
                logger.info(f"reshaped y data for plotting {title}")
            if len(c.shape) != 1:
                c = c.reshape((x.shape[0],))
                logger.info(f"reshaped c data for plotting {title}")
            data_shapes_ok = True

        else:
            data_shapes_ok = False
            logger.warning(f"cannot plot {title} because shapes do not match: x: {x.shape}, y: {y.shape}, c: {c.shape}")

    if data_shapes_ok:
        # #-----------------------------------------
        # if aspect_ratio=="equal":
        #     aspect_ratio = 1

        # if aspect_ratio=="auto":
        #     aspect_ratio = 2

        #-----------------------------------------
        # perform triangulation
        triang = tri.Triangulation(x, y)
        triangles = triang.triangles

        #-----------------------------------------
        # extract the maximum y values per x coordinate
        max_y_by_x = {}
        unique_x = np.unique(np.array(x))
        for x_value in unique_x:
            max_y_by_x[x_value] = np.max(y[np.where(x==x_value)])
        
        #-----------------------------------------
        # masking
        mask = []
        for points in triangles:
            #print points
            triang_is_not_inside = False
            #x_mean = np.mean(x[points])
            y_mean = np.mean(y[points])
            triang_is_not_inside = np.min([max_y_by_x[x_] for x_ in x[points]]) < y_mean
            mask.append(triang_is_not_inside)
        
        triang.set_mask(mask)
        
        #-----------------------------------------
        # Tricontour plot
        # Directly supply the unordered, irregularly spaced coordinates to tricontour.
        width = 8
        if aspect_ratio is None:
            used_aspect_ratio = (np.max(x)-np.min(x))/(np.max(y)-np.min(y))
        else: 
            used_aspect_ratio = aspect_ratio
        heigth =  width / used_aspect_ratio
        plt.close()
        fig, ax = plt.subplots(figsize=(width, heigth))

        # plot all triangles
        # ax.triplot(triang, color='red', lw=0.01)
        # ax.tricontour(x, y, z, levels=100, linewidths=0.1, colors='k')
        if center_cbar:
            max_abs = np.round(np.max(np.abs(c)), 1)
            vmin = -max_abs
            vmax = max_abs
            levels = np.linspace(vmin, vmax, 100)
            ticks_list = []
            ticks_list.append(vmin)
            ticks_list = [vmin] + list(np.arange(np.ceil(vmin)+1, np.floor(vmax),1)) + [vmax] 
        elif manual_clim is not None:
            
            if len(manual_clim)>2:
                levels = np.linspace(np.min(manual_clim), np.max(manual_clim), 512)
                # ticks_list = [vmin] + list(np.arange(np.ceil(vmin)+1, np.floor(vmax),tick_size)) + [vmax]
                ticks_list = manual_clim
            else:
                    
                cbar_range =  manual_clim[1]-manual_clim[0]
                # tick_size = np.round(cbar_range/5, decimals=1)
                decimals = int(np.ceil(np.abs(np.log10(cbar_range))))
                if (manual_clim[0] >= 0) and (manual_clim[0] < 1e-5):
                    vmin = 0  #np.np.floor(manual_clim[0])
                else:
                    vmin = np.round(manual_clim[0], decimals=decimals) - 1*10**(-decimals)  #np.np.floor(manual_clim[0])
                if (manual_clim[1] >= 0) and (manual_clim[1] < 1e-5):
                    vmax = 0
                else:
                    vmax = np.round(manual_clim[1], decimals=decimals) + 1*10**(-decimals)  #np.ceil(manual_clim[1])
                levels = np.linspace(vmin, vmax, 100)
                # ticks_list = [vmin] + list(np.arange(np.ceil(vmin)+1, np.floor(vmax),tick_size)) + [vmax]
                ticks_list = np.linspace(start=vmin, stop=vmax, num=5)
        else:
            vmin = np.min(c)
            vmax = np.max(c)
            levels = 200

        if levels_overwrite is not None:
            levels=levels_overwrite

        extend = "neither"
        if cmap is not None:
            if cmap._rgba_under is not None: 
                extend = 'min'
            if cmap._rgba_over is not None: 
                extend = 'max'
            if (cmap._rgba_under is not None) and (cmap._rgba_over is not None): 
                extend = 'both'

        cntr = ax.tricontourf(triang, c, levels=levels, cmap=cmap, extend=extend)#, norm=colors.Normalize(vmin=vmin, vmax=vmax)) , norm=colors.Normalize(vmin=vmin, vmax=vmax, clip=False)


        if aspect_ratio is not None:
            ax.set_aspect(aspect_ratio)
        else:
            ax.set_aspect('equal')
        #cntr.set_clim(-5,5)

        #ax.set_aspect(aspect_ratio)

        axins = inset_axes(ax, width = "5%", height = "100%", loc = 'lower left',
                   bbox_to_anchor = (1.02, 0., 1, 1), bbox_transform = ax.transAxes,
                   borderpad = 0)

        if center_cbar:
            cbar = fig.colorbar(cntr, cax=axins, ticks=ticks_list)#, shrink=1/aspect_ratio, pad=0.05)
        elif manual_clim is not None: 
            cbar = fig.colorbar(cntr, cax=axins, ticks=ticks_list)#, shrink=1/aspect_ratio, pad=0.05)
        else:
            cbar = fig.colorbar(cntr, cax=axins)#, shrink=1/aspect_ratio, pad=0.05)
        # ], extend='max')
        #caxis.min = vmin
        #ax2.plot(x, y, 'ko', ms=1)
        #ax2.set(xlim=(-2, 2), ylim=(-2, 2))

        title_mathmode = title.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
        ax.set_title(title_mathmode)
        x_label_mathmode = x_label.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
        ax.set_xlabel(x_label_mathmode)
        y_label_mathmode = y_label.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
        ax.set_ylabel(y_label_mathmode)
        c_label_mathmode = c_label.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
        cbar.ax.set_ylabel(c_label_mathmode)

        if custom_axis_limits is not None:
            ax.set_ylim(custom_axis_limits['y'])
            ax.set_xlim(custom_axis_limits['x'])
            #plt.tight_layout()

        if flip_y_axis:
            ax.set_ylim(ax.get_ylim()[::-1])

        draw_rectangle = True
        if draw_rectangle:
            R = 0.018
            ax.set_xlim((np.min(unique_x)-2, np.max(unique_x)))
            ax.set_ylim(-0.2)
            rect_error_calc = patches.Rectangle((-0.12/R, 0), (1+0.12)/R, 0.2/R, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect_error_calc)
            
        #plt.tight_layout()
        plt.savefig(f"{filepath}{filename}.png", dpi=300, bbox_inches='tight')
        # save_figure_in_multiple_formats(plt.gcf(), f"{filepath}{filename}", (5,6))
        # plt.savefig(f"{filepath}{filename}.pdf", dpi=300, bbox_inches='tight')
        # fig.set_figheight(5)
        # fig.set_figwidth(6)
        # plt.savefig(f"{filepath}{filename}.pgf", bbox_inches="tight")
        #plt.savefig(f"{filepath}{filename}.svg", bbox_inches='tight')

        logger.info("saved: " + f"{filepath}{filename}.png")

        if close_up_limits is not None:
            ax.set_ylim(close_up_limits['y'])
            ax.set_xlim(close_up_limits['x'])
            ax.set_aspect(close_up_aspect_ratio)
            if flip_y_axis:
                ax.set_ylim(ax.get_ylim()[::-1])
            
            ##################################################
            x_min = close_up_limits['x'][0] # np.min(x)
            x_max = close_up_limits['x'][1] # np.max(x)
            y_min = close_up_limits['y'][0] # np.min(y)
            y_max = close_up_limits['y'][1] # np.max(y)
            grid_x, grid_y = np.mgrid[x_min:x_max:1000j, y_min:y_max:1000j]
            coordinates = np.array([x, y]).T
            grid_u_x = griddata(coordinates, ux, (grid_x, grid_y), method="linear", fill_value=np.nan) 
            grid_u_y = griddata(coordinates, uy, (grid_x, grid_y), method="linear", fill_value=np.nan)  
            speed = np.sqrt(np.exp(ux) + np.exp(uy))
            grid_speed = griddata(coordinates, speed, (grid_x, grid_y), method="linear") 
            lw = 2*grid_speed / np.max(speed)
            # seed_points = np.array([0*np.ones(20), np.linspace(0,11,20)])
            rect_error_calc.remove()
            ax.streamplot(grid_x[:,0], grid_y[0,:], grid_u_x.T, grid_u_y.T, color=streamline_color, linewidth=lw) #, start_points=seed_points.T, integration_direction='forward')
            # Create a Rectangle patch
            rect = mlp.patches.Rectangle((x_min, 1.01), -x_min-0.01, y_max-1, linewidth=1, edgecolor='none', facecolor='white', zorder=2)
            # Add the patch to the Axes
            ax.add_patch(rect)
            #####################################################
            plt.savefig(f"{filepath}{filename}_closeup.png", dpi=300, bbox_inches='tight')
            #plt.savefig(f"{filepath}{filename}_closeup.pdf", dpi=300, bbox_inches='tight')
            #save_figure_in_multiple_formats(plt.gcf(), f"{filepath}{filename}_closeup", (5,6))

            fig.set_figheight(5)
            fig.set_figwidth(4)
            ax.set_title("")
            plt.savefig(f"{filepath}{filename}_closeup_small.png", dpi=300, bbox_inches='tight')
            # plt.savefig(f"{filepath}{filename}_closeup.pgf", bbox_inches="tight")
            # plt.savefig(f"{filepath}{filename}_closeup.pgf", format='pgf', bbox_inches='tight')
            #plt.savefig(f"{filepath}{filename}_closeup.pdf", dpi=300, bbox_inches='tight')
            #plt.savefig(f"{filepath}{filename}_closeup.svg", bbox_inches='tight')

            logger.info("saved: " + f"{filepath}{filename}_closeup.png")

        
        plt.close()



def plot_coordinates(x, y, z, source, export_path, show_all=False):
    """
    """
    logger.info("now plotting coordinates for: "+source)

    # CAUTION: the 3D projection always displays the z-axis from bottom to top
    # in our case I wanted the z-axis to go from back to front. 
    # Therefore I swapped the y and z data in the ax.scatter call and in the axis labels
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = plt.cm.get_cmap('RdYlBu')
    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')
    if not show_all:
        resolution = int(1e2) if x.shape[0] > 1e6 else int(1)
    else: 
        resolution = 1
    radius = y[::resolution]**2+ z[::resolution]**2
    ax.scatter(x[::resolution], z[::resolution], y[::resolution], c=radius, s=0.5, alpha=0.8, cmap=cm)
    ax.elev = 20# 20
    ax.azim = 230 # 240
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    if resolution == 1:
        ax.set_title(f"{source} - colored by radius")
    else: 
        ax.set_title(f"{source} - colored by radius\n (original data contains {x.shape[0]} points - for performance only {x[::resolution].shape[0]} are displayed)")
    plt.savefig(f"{export_path}{source} 3d.png", dpi=300)
    logger.info(f"saved {export_path}{source} 3d.png")
    plt.close()

    fig = plt.figure()
    plt.scatter(x,y, s=0.5)
    plt.xlabel("x")
    plt.xlabel("y")
    plt.title(f"{source} x over y")
    plt.savefig(f"{export_path}{source} x over y.png", dpi=300)
    logger.info(f"saved {export_path}{source} x over y.png")
    
    plt.close()
    fig = plt.figure()
    plt.scatter(x,z, s=0.5)
    plt.xlabel("x")
    plt.xlabel("z")
    plt.title(f"{source} x over z")
    plt.savefig(f"{export_path}{source} x over z.png", dpi=300)
    logger.info(f"saved {export_path}{source} x over z.png")

    fig = plt.figure()
    plt.close()
    plt.scatter(y,z, s=0.5)
    plt.xlabel("y")
    plt.xlabel("z")
    plt.title(f"{source} y over z")
    plt.savefig(f"{export_path}{source} y over z.png", dpi=300)
    logger.info(f"saved {export_path}{source} y over z.png")


def plot_inlet_velocity(inlet_position, x_inlet, rho_inlet, theta_inlet, ur_inlet, ut_inlet, ux_inlet, export_path, inlet_name="inlet"):
    """
    """

    size = 8

    logger.info(f"now plotting {inlet_name} velocities")
    fig = plt.figure()
    plt.scatter(ur_inlet, rho_inlet, s=size, marker="o", color="red")
    plt.plot(ur_inlet, rho_inlet, "k--")
    plt.xlabel(r"$u_r in m/s$")
    plt.ylabel(r"$\rho$ in m")
    plt.title(fr"ur over $\rho$ at {inlet_name} (at x={inlet_position})")
    plt.savefig(f"{export_path}ur over rho at {inlet_name}.png", dpi=300)

    fig = plt.figure()
    plt.scatter(ux_inlet, rho_inlet, s=size, marker="o", color="red")
    plt.plot(ux_inlet, rho_inlet, "k--")
    plt.xlabel(r"$u_x in m/s$")
    plt.ylabel(r"$\rho$ in m")
    plt.title(fr"$u_x$ over $\rho$ at {inlet_name} (at x={inlet_position})")
    plt.savefig(f"{export_path}ux over rho at {inlet_name}.png", dpi=300)

    fig = plt.figure()
    plt.scatter(ut_inlet, rho_inlet, s=size, marker="o", color="red")
    plt.plot(ut_inlet, rho_inlet, "k--")
    plt.xlabel(r"$u_t in m/s$")
    plt.ylabel(r"$\rho$ in m")
    plt.title(fr"ut over $\rho$ at {inlet_name} (at x={inlet_position})")
    plt.savefig(f"{export_path}ut over rho at {inlet_name}.png", dpi=300)

    fig = plt.figure()
    plt.scatter(x_inlet, rho_inlet, s=size, marker="o", color="red")
    plt.plot(x_inlet, rho_inlet, "k--")
    plt.xlabel(r"x")
    plt.ylabel(r"$\rho$ in m")
    plt.title(fr"x over $\rho$ at {inlet_name} (at x={inlet_position})")
    plt.savefig(f"{export_path}x over rho at {inlet_name}.png", dpi=300)

    fig = plt.figure()
    plt.scatter(x_inlet, theta_inlet, s=size, marker="o", color="red")
    plt.plot(x_inlet, theta_inlet, "k--")    
    plt.xlabel("x")
    plt.ylabel(r"$\theta$ in m")
    plt.title(fr"x over $\theta$ at {inlet_name} (at x={inlet_position})")
    plt.savefig(f"{export_path}x over theta at {inlet_name}.png", dpi=300)

    plt.close()


def plot_interpolation_as_scatterplot(interp_values, interp_x_coordinates, interp_y_coordinates, input_coord, input_data, title, filename, path, logger):
    """
    """
    # specify x and y coordinates of boundaries to make visualize boundary conditions points
    y_bc_lines = [0, 0.018, 1.4]
    x_bc_lines = [-0.12, 0, 2.8]

    logger.info(f"now plotting {filename}")
    value_index = 0
    sc_size = 200
    pad = (np.max(interp_x_coordinates)-np.min(interp_x_coordinates))*0.5

    vmin = np.min(interp_values[:,:,value_index]) #np.min(input_data[:,value_index])
    vmax = np.max(interp_values[:,:,value_index]) #np.max(input_data[:,value_index])

    fig, axs = plt.subplots(figsize=(6,6))
    # plot scatter with original data
    #sc = axs.scatter(input_coord[:,0], input_coord[:,1], c=input_data[:,value_index], s=sc_size, cmap="RdBu_r", edgecolors='black',vmin=vmin, vmax=vmax)
    axs.add_patch(mlp.patches.Rectangle(
            (np.min(interp_x_coordinates), np.min(interp_y_coordinates)),
            (np.max(interp_x_coordinates)-np.min(interp_x_coordinates)),
            (np.max(interp_y_coordinates)-np.min(interp_y_coordinates)),
            edgecolor='k',
            facecolor='none',
            lw=1))
    
    axs.set_aspect('equal')
    #axs.set_title("input data")
    axs.set_xlabel("x")
    axs.set_ylabel("y")
    axs.set_xlim([np.min(interp_x_coordinates)-pad, np.max(interp_x_coordinates)+pad])
    axs.set_ylim([np.min(interp_y_coordinates)-pad, np.max(interp_y_coordinates)+pad])

    for _y in y_bc_lines:
        axs.plot(axs.get_xlim(), (_y, _y), c="k")
    for _x in x_bc_lines:
        axs.plot((_x, _x), axs.get_ylim(), c="k")

    im = axs.imshow(interp_values[:,:,value_index],
                        extent=(np.min(interp_x_coordinates),np.max(interp_x_coordinates), np.min(interp_y_coordinates), np.max(interp_y_coordinates)),
                        cmap="RdBu_r", origin="lower", interpolation='none',
                        vmin=vmin, vmax=vmax)

    cbar1 = fig.colorbar(im, ax=axs, location='right', anchor=(0.4, 0.4), shrink=0.6)#, ticks=ticks_list)
    plt.xticks(rotation=70, ha="right")
    axs.set_title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(path, filename), bbox_inches="tight", dpi=200)
    logger.info(f"saved {os.path.join(path, filename)}")
    plt.close()



def plot_interpolation_new(orig_grid, orig_data_raw, interp_values, interp_x_coordinates, interp_y_coordinates, input_coord, input_data, title, filename, path, logger):
    """
    """
    plt.close()
    # specify x and y coordinates of boundaries to make visualize boundary conditions points
    y_bc_lines = [0, 0.018, 1.4]
    x_bc_lines = [-0.12, 0, 2.8]

    logger.info(f"now plotting {filename}")
    value_index = 0
    sc_size = 200
    pad = (np.max(interp_x_coordinates)-np.min(interp_x_coordinates))*0.2
    orig_data = orig_data_raw[:,0]

    vmin = np.min(interp_values[:,:,value_index]) #np.min(input_data[:,value_index])
    vmax = np.max(interp_values[:,:,value_index]) #np.max(input_data[:,value_index])


    fig, axs = plt.subplots(figsize=(6,6))
    # plot scatter with original data
    #sc = axs.scatter(input_coord[:,0], input_coord[:,1], c=input_data[:,value_index], s=sc_size, cmap="RdBu_r", edgecolors='black',vmin=vmin, vmax=vmax)


    # extract he maximum y values per x coordinate
    triang = tri.Triangulation(orig_grid[:,0], orig_grid[:,1])
    x = orig_grid[:,0]
    y = orig_grid[:,1]
    max_y_by_x = {}
    unique_x = np.unique(np.array(x))
    for x_value in unique_x:
        max_y_by_x[x_value] = np.max(y[np.where(x==x_value)])
    # masking
    mask = []
    for points in triang.triangles:
        #print points
        triang_is_not_inside = False
        #x_mean = np.mean(x[points])
        y_mean = np.mean(y[points])
        triang_is_not_inside = np.min([max_y_by_x[x_] for x_ in x[points]]) < y_mean
        mask.append(triang_is_not_inside)
    
    triang.set_mask(mask)
    imcontourf = axs.tricontourf(triang, orig_data, levels=8000,  cmap="RdBu_r", zorder=-1,
                        vmin=vmin, vmax=vmax)

    axs.add_patch(mlp.patches.Rectangle(
        (np.min(interp_x_coordinates), np.min(interp_y_coordinates)),
        (np.max(interp_x_coordinates)-np.min(interp_x_coordinates)),
        (np.max(interp_y_coordinates)-np.min(interp_y_coordinates)),
        edgecolor='k',
        facecolor='none',
        lw=1))
    
    axs.set_aspect('equal')
    #axs.set_title("input data")
    axs.set_xlabel("x")
    axs.set_ylabel("y")
    axs.set_xlim([np.min(interp_x_coordinates)-pad, np.max(interp_x_coordinates)+pad])
    axs.set_ylim([np.min(interp_y_coordinates)-pad, np.max(interp_y_coordinates)+pad])

    # for _y in y_bc_lines:
    #     axs.plot(axs.get_xlim(), (_y, _y), c="k")
    # for _x in x_bc_lines:
    #     axs.plot((_x, _x), axs.get_ylim(), c="k")

    im = axs.imshow(interp_values[:,:,value_index],
                        extent=(np.min(interp_x_coordinates),np.max(interp_x_coordinates), np.min(interp_y_coordinates), np.max(interp_y_coordinates)),
                        cmap="RdBu_r", origin="lower", interpolation='none',
                        vmin=vmin, vmax=vmax)
                        # Major ticks
    # ax = plt.gca()
    # ax.set_xticks(np.arange(np.min(interp_x_coordinates), np.max(interp_x_coordinates), 1))
    # ax.set_yticks(np.arange(np.min(interp_y_coordinates), np.max(interp_y_coordinates), 1))
    # Minor ticks
    #ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    #ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
    # Gridlines based on minor ticks
    #ax.grid(which='minor', color='k', linestyle='-', linewidth=2)

    cbar1 = fig.colorbar(im, ax=axs, location='right', anchor=(0.4, 0.4), shrink=0.6, label="$u_x$ in $m/s$")#, ticks=ticks_list)
    
    #cbar2 = fig.colorbar(imcontourf, ax=axs, location='right', anchor=(0.4, 0.4), shrink=0.6)#, ticks=ticks_list)
    plt.xticks(rotation=70, ha="right")
    axs.set_title(title)
    plt.tight_layout()
    plt.xlabel("x")
    plt.ylabel("r")
    #plt.clabel(r"$u_x$")
    fig.savefig(os.path.join(path, filename), bbox_inches="tight", dpi=200)
    logger.info(f"saved {os.path.join(path, filename)}")
    plt.close()

def plot_interpolated_cell_batch(interpolated_data_raw, x_coordinates, y_coordinates, orig_grid, orig_data_raw, title, filename, path, logger):
    """
    """

    logger.info(f"now plotting {filename}")

    pad = (np.max(x_coordinates) - np.min(x_coordinates))/2

    interpolated_data = interpolated_data_raw[:,:,0]
    orig_data = orig_data_raw[:,0]

    point_inside_x = np.logical_and((orig_grid[:,0] >= np.min(x_coordinates)-pad), (orig_grid[:,0] <= np.max(x_coordinates)+pad))
    point_inside_y = np.logical_and((orig_grid[:,1] >= np.min(y_coordinates)-pad), (orig_grid[:,1] <= np.max(y_coordinates)+pad))
    point_in_cell_batch = np.logical_and(point_inside_x, point_inside_y)
    cropped_grid = orig_grid[point_in_cell_batch]
    cropped_data = orig_data[point_in_cell_batch]

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9,4))
    # interpolated data
    im = axs[0].imshow(interpolated_data, cmap="RdBu_r", origin="lower", interpolation='none')
    axs[0].set_title("interpolated cell batch")
    cbar1 = fig.colorbar(im, ax=axs[0], location='right', anchor=(0.4, 0.4), shrink=0.4)#, ticks=ticks_list)
    # try to plot triangulation
    try:
        triang = tri.Triangulation(cropped_grid[:,0], cropped_grid[:,1])
        cntr = axs[1].tricontourf(triang, cropped_data, levels=200, cmap="RdBu_r")
        cbar2 = fig.colorbar(cntr, ax=axs[1], location='right', anchor=(0.4, 0.4), shrink=0.4)
    except:
        logger.warning("triangulation failed")
    # plot scatter with original data
    axs[1].scatter(cropped_grid[:,0], cropped_grid[:,1], c=cropped_data, cmap="RdBu_r", edgecolors='black')
    axs[1].set_xlim([np.min(x_coordinates), np.max(x_coordinates)])
    axs[1].set_ylim([np.min(y_coordinates), np.max(y_coordinates)])
    axs[1].set_aspect('equal')
    axs[1].set_title("original")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")

    # plot position of cell batch

    # extract he maximum y values per x coordinate
    triang = tri.Triangulation(orig_grid[:,0], orig_grid[:,1])
    x = orig_grid[:,0]
    y = orig_grid[:,1]
    max_y_by_x = {}
    unique_x = np.unique(np.array(x))
    for x_value in unique_x:
        max_y_by_x[x_value] = np.max(y[np.where(x==x_value)])
    # masking
    mask = []
    for points in triang.triangles:
        #print points
        triang_is_not_inside = False
        #x_mean = np.mean(x[points])
        y_mean = np.mean(y[points])
        triang_is_not_inside = np.min([max_y_by_x[x_] for x_ in x[points]]) < y_mean
        mask.append(triang_is_not_inside)
    
    triang.set_mask(mask)
    cntr2 = axs[2].tricontourf(triang, orig_data, levels=200,  cmap="RdBu_r")
    cbar3 = fig.colorbar(cntr2, ax=axs[2], location='right', anchor=(0.4, 0.4), shrink=0.4)
    width = (np.max(x_coordinates)-np.min(x_coordinates))
    height = (np.max(y_coordinates)-np.min(y_coordinates))
    rect = mlp.patches.Rectangle((x_coordinates[0], y_coordinates[0]), width, height, facecolor="black")
    axs[2].add_patch(rect)
    padding = width*10
    #axs[2].scatter(orig_data[:,0], orig_data[:,1])
    axs[2].set_xlim([np.mean(x_coordinates)-padding, np.mean(x_coordinates)+padding])
    axs[2].set_ylim([np.mean(y_coordinates)-padding, np.mean(y_coordinates)+padding])
    axs[2].set_aspect('equal')
    axs[2].set_title("position")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")

    #plt.subplots_adjust(wspace=0.4, hspace=-0.1)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(os.path.join(path, filename), bbox_inches="tight", dpi=200)
    logger.info(f"saved {os.path.join(path, filename)}")
    plt.close()

def plot_cell_batch_coverage(cell_batches, orig_grid, orig_data, title, filename, path, logger):
    """
    plot_cell_batch_coverage(cell_batches, self.BlockMesh_coordinates_2D, values, "comparison of cell batch interpolation to original data",
                                            f"cell_batch_{cell_batch_counter}.png",
                                            self.path_to_postprocessing_folder, self.logger)
    """

    logger.info(f"now plotting {filename}")

    fig, ax = plt.subplots(figsize=(6,4))
    # extract he maximum y values per x coordinate
    triang = tri.Triangulation(orig_grid[:,0], orig_grid[:,1])
    x = orig_grid[:,0]
    y = orig_grid[:,1]
    max_y_by_x = {}
    unique_x = np.unique(np.array(x))
    for x_value in unique_x:
        max_y_by_x[x_value] = np.max(y[np.where(x==x_value)])
    # masking
    mask = []
    for points in triang.triangles:
        #print points
        triang_is_not_inside = False
        #x_mean = np.mean(x[points])
        y_mean = np.mean(y[points])
        triang_is_not_inside = np.min([max_y_by_x[x_] for x_ in x[points]]) < y_mean
        mask.append(triang_is_not_inside)
    
    triang.set_mask(mask)

    cntr = ax.tricontourf(triang, orig_data[:,0], levels=200,  cmap="RdBu_r")
    cbar = fig.colorbar(cntr, ax=ax, location='right', anchor=(0.4, 0.4), shrink=0.4)

    for cell_batch in cell_batches:
        x_batch = cell_batch[0]
        y_batch = cell_batch[1]
        width = (np.max(x_batch)-np.min(x_batch))
        height = (np.max(y_batch)-np.min(y_batch))
        rect = mlp.patches.Rectangle((x_batch[0], y_batch[0]), width, height, facecolor="black")
        ax.add_patch(rect)
    
    ax.set_aspect('equal')
    ax.set_title("position")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(path, filename), bbox_inches="tight", dpi=200)
    logger.info(f"saved {os.path.join(path, filename)}")
    plt.close()



def plot_line_probes(probe_x_coordinate : float, coordinates : list, fields : list, names : list, field_parameter_name : str, ylabel : str, title_mathmode : str, export_path : str, name_template : str, close_up_value : float):
    """
    
    """

    print("plotting line probes")


    colors = ["blue", "red", "green", "cyan", "purple"]
    
    for do_closeup in [False, True]:

        fig = plt.figure()
        
        for coordinate, field, name, i in zip(coordinates, fields, names, range(len(names))):

            x = coordinate[:, 0]
            y = coordinate[:, 1]

            x_distance_to_probe = np.abs(x-probe_x_coordinate)

            indexes_of_probe = np.where((x_distance_to_probe == np.min(x_distance_to_probe)))[0]
            
            values = field[indexes_of_probe]
            y_coords_of_values = y[indexes_of_probe]

            size=10
            plt.scatter(values, y_coords_of_values, label=name, s=size, marker="o", color=colors[i])
            plt.plot(values, y_coords_of_values, color=colors[i])
            
        
        if do_closeup:
            #plt.yscale('log')
            plt.ylim((0, np.min([close_up_value, 1.1*np.max(y_coords_of_values)])))
            #plt.xscale('symlog')
            closeup_fname = "_closeup"
        else:
            closeup_fname = ""
        plt.legend(bbox_to_anchor=(0.93,-0.23), loc="lower right", 
                bbox_transform=fig.transFigure)
        field_parameter_name_mathmode = field_parameter_name.replace(' ','\/')
        plt.xlabel(rf"${field_parameter_name_mathmode}$")
        plt.ylabel(ylabel)
        plt.grid()
        plt.title(rf"${title_mathmode}$")
        plt.savefig(f"{export_path}{name_template}_x{probe_x_coordinate}{closeup_fname}.png", dpi=300, bbox_inches="tight")



def plot_line_probes_with_reference(probe_x_coordinate : float, coordinates : list, fields : list, reference_field, names : list, field_parameter_name : str, export_path = str, name_template = str, percentage_mode : Boolean = False):
    """
    
    """

    print("plotting line probes")


    colors = ["blue", "red", "green", "cyan", "purple"]
    
    for do_closeup in [False, True]:

        fig = plt.figure()
        
        for coordinate, field, name, i in zip(coordinates, fields, names, range(len(names))):

            x = coordinate[:, 0]
            y = coordinate[:, 1]

            x_distance_to_probe = np.abs(x-probe_x_coordinate)

            indexes_of_probe = np.where((x_distance_to_probe == np.min(x_distance_to_probe)))[0]
            
            if percentage_mode:
                values = 100*(field[indexes_of_probe] - reference_field[indexes_of_probe])/ reference_field[indexes_of_probe]
            else:
                values = field[indexes_of_probe] - reference_field[indexes_of_probe]
            y_coords_of_values = y[indexes_of_probe]

            size=10
            plt.scatter(values, y_coords_of_values, label=name, s=size, marker="o", color=colors[i+1])
            plt.plot(values, y_coords_of_values, color=colors[i+1])
        
        if do_closeup:
            #plt.yscale('log')
            plt.ylim((0, 0.2))
            closeup_fname = "_closeup"
        else:
            closeup_fname = ""
        plt.axvline(x=0.0, color=colors[0])
        plt.legend(bbox_to_anchor=(0.93,-0.21), loc="lower right", 
                bbox_transform=fig.transFigure)
        plt.xlabel(f"{field_parameter_name}")
        plt.ylabel(f"r in m")
        plt.title(f"{name_template} at x={probe_x_coordinate} m")
        plt.savefig(f"{export_path}{name_template}_x{probe_x_coordinate}{closeup_fname}.png", dpi=300, bbox_inches="tight")


def plot_spreading_rate(coordinates : list, fields : list, names : list, colors : list, 
                xlabel : str, ylabel : str, title_mathmode : str, export_path : str, name_template : str,
                xlim : float, ylim : float, smoothen = False):

    print("plotting spreading rate")

    fig, ax = plt.subplots(figsize=(4.5,2.5))
    R = 0.018 
    D = 0.035
    unique_x_coords = np.sort(np.unique(coordinates[0]))
    ax.fill_between(unique_x_coords/D, 0.086 * np.ones(unique_x_coords.shape), 0.096 * np.ones(unique_x_coords.shape), alpha=0.3, color="grey")
    
    for coordinate, field, name, i in zip(coordinates[::-1], fields[::-1], names[::-1], range(len(names))):
        
        x = coordinate[:, 0]/D
        y = coordinate[:, 1]/D
        #x_sorting_ids = np.argsort(x)
        unique_x_coords = np.sort(np.unique(x))
        spreading_rates = []
        for x_coordinate in unique_x_coords:
            indexes_of_x_probe = np.where((x == x_coordinate))[0]
            velocity_values = field[indexes_of_x_probe]     
            y_values =  y[indexes_of_x_probe]   

            index_of_centerline = np.where((y_values == np.min(y_values)))[0]
            y_value_of_centerline = y_values[index_of_centerline]
            centerline_velocity =  velocity_values[index_of_centerline][0]
            half_centerline_velocity = centerline_velocity/2

            velocity_difference_to_half_centerline_velocity = np.abs(velocity_values-half_centerline_velocity)
            index_of_half_centerline_velocity = np.where((velocity_difference_to_half_centerline_velocity == np.min(velocity_difference_to_half_centerline_velocity)))[0]            
            y_value_of_half_centerline_velocity = y_values[index_of_half_centerline_velocity]
            spreading_rate = y_value_of_half_centerline_velocity/x_coordinate
            spreading_rates.append(spreading_rate[0])

        if smoothen:
            spl = make_interp_spline(unique_x_coords[::2], spreading_rates[::2], k=1)  # type: BSpline
            new_x_coords = np.linspace(0, np.max(unique_x_coords), num=200)
            spreading_rates_new = spl(new_x_coords)
            ax.plot(new_x_coords, spreading_rates_new, color=colors[i], linewidth=1.2)
        else:
            ax.scatter(unique_x_coords, spreading_rates, s=1, marker="o", color=colors[i])
            ax.plot(unique_x_coords, spreading_rates, color=colors[i], linewidth=1.2)


    if xlim is not None:
        ax.set_xlim((0, xlim))
    if ylim is not None:
        ax.set_ylim((0, ylim))

    # manually define a new patch 
    experimental_data_name= "experimental data"#"experimental data for round jet flow (0.086-0.096)"

    names_for_legend = names.copy()
    names_for_legend.insert(0, experimental_data_name)
    # lg = plt.legend(names_for_legend[::-1], bbox_to_anchor=(0.89,-0.62), loc="lower right", 
    #         bbox_transform=fig.transFigure)
    lg = plt.legend(names_for_legend[::-1], loc="best", frameon=False)
    xlabel_mathmode = xlabel.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
    ylabel_mathmode = ylabel.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 

    plt.xlabel(xlabel_mathmode)
    plt.ylabel(ylabel_mathmode)
    # plt.grid()
    title_mathmode_modified = title_mathmode.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") # "axial velocities\n__MATHMODESTART__u_{{ref}} = max(u_{{x,LES}}) = {ref_velocity:.3f} m/s__MATHMODEEND__"
    plt.title(title_mathmode_modified)
    plt.savefig(f"{export_path}{name_template}.png", dpi=300, bbox_inches="tight")

    plt.title("")
    plt.savefig(f"{export_path}{name_template}.pdf", dpi=300, bbox_inches="tight")

    # paper like version
    # fig.set_figheight(2)
    # fig.set_figwidth(5)
    # lg.remove()
    # plt.legend(names[::-1], fancybox=False, framealpha=1.0)
    # plt.savefig(f"{export_path}{name_template}_small.pdf", dpi=300, bbox_inches="tight")
    
    plt.close()


def plot_spreading_rate_2(coordinates : list, fields : list, names : list, colors : list, 
                xlabel : str, ylabel : str, title_mathmode : str, export_path : str, name_template : str,
                xlim : float, ylim : float, smoothen = False):

    print("plotting spreading rate2")
    fig, ax = plt.subplots(figsize=(4.5,2.5))
    unique_x_coords = np.sort(np.unique(coordinates[0]))
    ax.fill_between(unique_x_coords, 0.086 * np.ones(unique_x_coords.shape), 0.096 * np.ones(unique_x_coords.shape), alpha=0.3, color="grey")
    for coordinate, field, name, i in zip(coordinates[::-1], fields[::-1], names[::-1], range(len(names))):
        x = coordinate[:, 0]

        if smoothen:
            spl = make_interp_spline(x[::2], field[::2], k=1)  # type: BSpline
            new_x_coords = np.linspace(0, np.max(unique_x_coords), num=200)
            spreading_rates_new = spl(new_x_coords)
            ax.plot(new_x_coords, spreading_rates_new, color=colors[i], linewidth=1.2)
        else:
            ax.scatter(x, field, s=1, marker="o", color=colors[i])
            ax.plot(x, field, color=colors[i], linewidth=1.2)

    if xlim is not None:
        ax.set_xlim((0, xlim))
    if ylim is not None:
        ax.set_ylim((0, ylim))

    # manually define a new patch 
    experimental_data_name= "experimental data (Wilcox)"#"experimental data for round jet flow (0.086-0.096)"

    names_for_legend = names.copy()
    names_for_legend.insert(0, experimental_data_name)
    # lg = plt.legend(names_for_legend[::-1], bbox_to_anchor=(0.89,-0.62), loc="lower right", 
    #         bbox_transform=fig.transFigure)
    lg = plt.legend(names_for_legend[::-1], loc="best", frameon=False)
    xlabel_mathmode = xlabel.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
    ylabel_mathmode = ylabel.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 

    plt.xlabel(xlabel_mathmode)
    plt.ylabel(ylabel_mathmode)
    # plt.grid()
    title_mathmode_modified = title_mathmode.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") # "axial velocities\n__MATHMODESTART__u_{{ref}} = max(u_{{x,LES}}) = {ref_velocity:.3f} m/s__MATHMODEEND__"
    plt.title(title_mathmode_modified)
    plt.savefig(f"{export_path}{name_template}2.png", dpi=300, bbox_inches="tight")
    plt.title("")
    plt.savefig(f"{export_path}{name_template}2.pdf", dpi=300, bbox_inches="tight")
   
    plt.close()


def plot_axial_line_plot_with_self_similarity(probe_coordinate : float, coordinates : list, fields : list, names : list, 
                xlabel : str, ylabel : str, title_mathmode : str, export_path : str, name_template : str,
                xlim : float, ylim : float):

    print("plotting multiprobe_line_plot")
    colors = ["red","blue",  "green", "cyan", "purple"]
    colors[len(coordinates)-1] = "black"

    fig = plt.figure(figsize=(4.5,2.5))
    #plt.rcParams['font.size'] = '16'
    
    for coordinate, field, name, i in zip(coordinates[::-1], fields[::-1], names[::-1], range(len(names))):
        
        x = coordinate[:, 0]
        y = coordinate[:, 1]

        y_distance_to_probe = np.abs(y-probe_coordinate)

        indexes_of_probe = np.where((y_distance_to_probe == np.min(y_distance_to_probe)))[0]            
        values = field[indexes_of_probe]
        x_coords_of_values = x[indexes_of_probe]

        # # area under line
        # inylim = np.where(y_coords_of_values <= ylim)
        # area = np.trapz(values[inylim],y_coords_of_values[inylim])

        # plt.scatter(x_coords_of_values, values, s=2, marker="o", color=colors[i])
        plt.plot(x_coords_of_values, values, color=colors[i], linewidth=1.2)
        # plt.text(1.1*np.mean(values)+x_anchor, 0.9*ylim-0.3*(i/ylim), f"{area:.4f}", c=colors[i])
        # print(f"{area:.4f}")
    self_similar_name= "reference value (Ball et al.)" # "axial mean velocity decay in self-similar region (decay constant = 6)"
    decay_constants = [6]
    for d in decay_constants:
        self_similarity_curve = 1/d * (x_coords_of_values)
        plt.plot(x_coords_of_values, self_similarity_curve, color="grey", linewidth=1.2, linestyle="dashed")
    
    if xlim is not None:
        plt.xlim((0, xlim))
    
    if ylim is not None:
        plt.ylim((0, ylim))
    
    names_for_legend = names.copy()
    names_for_legend.insert(0,self_similar_name)
    # lg = plt.legend(names_for_legend[::-1], bbox_to_anchor=(0.89,-0.62), loc="lower right", 
    #         bbox_transform=fig.transFigure)
    lg = plt.legend(names_for_legend[::-1], loc="best", frameon=False)

    xlabel_mathmode = xlabel.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
    ylabel_mathmode = ylabel.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 

    plt.xlabel(xlabel_mathmode)
    plt.ylabel(ylabel_mathmode)
    # plt.grid()
    title_mathmode_modified = title_mathmode.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") # "axial velocities\n__MATHMODESTART__u_{{ref}} = max(u_{{x,LES}}) = {ref_velocity:.3f} m/s__MATHMODEEND__"
    plt.title(title_mathmode_modified)
    plt.savefig(f"{export_path}{name_template}.png", dpi=300, bbox_inches="tight")

    plt.title("")
    plt.savefig(f"{export_path}{name_template}.pdf", dpi=300, bbox_inches="tight")

    # paper like version
    # fig.set_figheight(2)
    # fig.set_figwidth(5)
    # lg.remove()
    # plt.legend(names[::-1], fancybox=False, framealpha=1.0)
    # plt.savefig(f"{export_path}{name_template}_small.pdf", dpi=300, bbox_inches="tight")
    
    plt.close()


def plot_axial_line_plot(probe_coordinate : float, coordinates : list, fields : list, names : list, 
                xlabel : str, ylabel : str, title_mathmode : str, export_path : str, name_template : str,
                xlim : float):

    print("plotting multiprobe_line_plot")
    colors = ["blue", "red", "green", "cyan", "purple"]
    colors[len(coordinates)-1] = "black"

    fig = plt.figure(figsize=(7,2.5))
    #plt.rcParams['font.size'] = '16'
    
    for coordinate, field, name, i in zip(coordinates[::-1], fields[::-1], names[::-1], range(len(names))):
        
        x = coordinate[:, 0]
        y = coordinate[:, 1]

        y_distance_to_probe = np.abs(y-probe_coordinate)

        indexes_of_probe = np.where((y_distance_to_probe == np.min(y_distance_to_probe)))[0]            
        values = field[indexes_of_probe]
        x_coords_of_values = x[indexes_of_probe]

        # # area under line
        # inylim = np.where(y_coords_of_values <= ylim)
        # area = np.trapz(values[inylim],y_coords_of_values[inylim])

        plt.scatter(x_coords_of_values, values, s=2, marker="o", color=colors[i])
        plt.plot(x_coords_of_values, values, color=colors[i], linewidth=0.5)
        # plt.text(1.1*np.mean(values)+x_anchor, 0.9*ylim-0.3*(i/ylim), f"{area:.4f}", c=colors[i])
        # print(f"{area:.4f}")
    
    if xlim is not None:
        plt.xlim((0, xlim))

    lg = plt.legend(names[::-1], bbox_to_anchor=(0.89,-0.4), loc="lower right", 
            bbox_transform=fig.transFigure)

    xlabel_mathmode = xlabel.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
    ylabel_mathmode = ylabel.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 

    plt.xlabel(xlabel_mathmode)
    plt.ylabel(ylabel_mathmode)
    plt.grid()
    title_mathmode_modified = title_mathmode.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") # "axial velocities\n__MATHMODESTART__u_{{ref}} = max(u_{{x,LES}}) = {ref_velocity:.3f} m/s__MATHMODEEND__"
    plt.title(title_mathmode_modified)
    plt.savefig(f"{export_path}{name_template}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{export_path}{name_template}.pdf", dpi=300, bbox_inches="tight")

    # paper like version
    fig.set_figheight(2)
    fig.set_figwidth(5)
    lg.remove()
    plt.legend(names[::-1], fancybox=False, framealpha=1.0)
    plt.savefig(f"{export_path}{name_template}_small.pdf", dpi=300, bbox_inches="tight")
    
    plt.close()


def plot_singleprobe_line_plot(probe_coordinate : float, coordinates : list, fields : list, names : list, 
                xlabel : str, ylabel : str, title_mathmode : str, export_path : str, name_template : str,
                ylim : float):

    print("plotting multiprobe_line_plot")
    colors = ["blue", "red", "green", "cyan", "purple"]
    colors[len(coordinates)-1] = "black"

    fig = plt.figure(figsize=(3,3))

    for coordinate, field, name, i in zip(coordinates[::-1], fields[::-1], names[::-1], range(len(names))):
        
        x_anchor = probe_coordinate
        x = coordinate[:, 0]
        y = coordinate[:, 1]

        x_distance_to_probe = np.abs(x-probe_coordinate)

        indexes_of_probe = np.where((x_distance_to_probe == np.min(x_distance_to_probe)))[0]            
        values = field[indexes_of_probe]
        y_coords_of_values = y[indexes_of_probe]

        plt.scatter(values, y_coords_of_values, s=1, marker="o", color=colors[i])
        plt.plot(values, y_coords_of_values, color=colors[i], linewidth=0.8)     
    
    plt.ylim((0, np.min([ylim, 1.1*np.max(y_coords_of_values)])))
    lg = plt.legend(names[::-1], bbox_to_anchor=(0.895,-0.50), loc="lower right", 
            bbox_transform=fig.transFigure)
    xlabele_mathmode = xlabel.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
    plt.xlabel(xlabele_mathmode)
    plt.ylabel(ylabel)
    
    title_mathmode_modified = title_mathmode.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") # "axial velocities\n__MATHMODESTART__u_{{ref}} = max(u_{{x,LES}}) = {ref_velocity:.3f} m/s__MATHMODEEND__"
    plt.grid(zorder=-100)
    plt.title(title_mathmode_modified)
    plt.savefig(f"{export_path}{name_template}.png", dpi=300, bbox_inches="tight")

    # plt.title("")
    # plt.savefig(f"{export_path}{name_template}.pdf", dpi=300, bbox_inches="tight")

    print(f"plotted: {export_path}{name_template}.png")


def new_separate_line_plot(probe_coordinates : list, probe_names : list, coordinates : list, fields : list, names : list, colors : list,
                xlabel : str, ylabel : str, title_mathmode : str, export_path : str, name_template : str,
                xlim : Union[None, float, list], ylim : Union[None, float, list] ):
    from matplotlib.lines import Line2D
    plt.close()
    nprobes = len(probe_coordinates)
    fig, axes = plt.subplots(nrows=1, ncols=nprobes, figsize=(2.5*nprobes,2.5))

    # probe_markers = ["d", "o", "x", "s", "+", "<", ">"]
    
    for probe_coordinate, probe_name, i in zip(probe_coordinates, probe_names, range(nprobes)):
        for coordinate, field, name, color in zip(coordinates, fields, names, colors):
            x = coordinate[:, 0]
            y = coordinate[:, 1]
            x_distance_to_probe = np.abs(x-probe_coordinate)
            indexes_of_probe = np.where((x_distance_to_probe == np.min(x_distance_to_probe)))[0]  
            values = field[indexes_of_probe]
            y_coords_of_values = y[indexes_of_probe]
            axes[i].plot(y_coords_of_values, values, color=color, linewidth=1.2) # marker=probe_marker, markevery=3
            xlabel_mathmode = xlabel.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
            ylabel_mathmode = ylabel.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
            axes[i].set_xlabel(xlabel_mathmode)
            # axes[i].set_ylabel(ylabel_mathmode)
            axes[i].set_title(probe_name)
            if ylim is not None:
                if type(ylim) is list:
                    axes[i].set_xlim((0, xlim[i]))
                else:
                    axes[i].set_xlim((0, xlim))
            if xlim is not None:
                if type(xlim) is list:
                    axes[i].set_xlim((0, xlim[i]))
                else:
                    axes[i].set_xlim((0, xlim))
    axes[0].set_ylabel(ylabel_mathmode)
    handles = []
    # for name, probe_marker in zip(names, probe_markers):
    #     handles.append(Line2D([0], [0], label=name, color="black", linestyle=" ", marker=probe_marker))
    for name, color in zip(names, colors):
        handles.append(Line2D([0], [0], label=name, color=color, linestyle="solid"))

    axes[0].legend(handles=handles, loc="best", frameon=False)

    # plt.legend(bbox_to_anchor=(0.895,-0.50), loc="lower right", 
    #         bbox_transform=fig.transFigure)
    title_mathmode_modified = title_mathmode.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") # "axial velocities\n__MATHMODESTART__u_{{ref}} = max(u_{{x,LES}}) = {ref_velocity:.3f} m/s__MATHMODEEND__"
    # plt.grid(zorder=-100)
    plt.suptitle(title_mathmode_modified)
    fig.tight_layout()
    plt.savefig(f"{export_path}{name_template}.png", dpi=300, bbox_inches="tight")

    # plt.title("")
    # plt.savefig(f"{export_path}{name_template}.pdf", dpi=300, bbox_inches="tight")

    print(f"plotted: {export_path}{name_template}.png")

def new_line_plot(probe_coordinates : list, probe_names : list, coordinates : list, fields : list, deltas: list, names : list, colors : list,
                xlabel : str, ylabel : str, title_mathmode : str, export_path : str, name_template : str,
                xlim : Union[None, float], ylim : Union[None, float] ):
    from matplotlib.lines import Line2D
    plt.close()
    fig, ax = plt.subplots(figsize=(4,4))    
    probe_markers = ["d", "o", "x", "s", "+", "<", ">"]
    for coordinate, field, delta, name, probe_marker in zip(coordinates, fields, deltas, names, probe_markers):
        for probe_coordinate, color in zip(probe_coordinates, colors):
            x = coordinate[:, 0]
            y = coordinate[:, 1]
            x_distance_to_probe = np.abs(x-probe_coordinate)
            indexes_of_probe = np.where((x_distance_to_probe == np.min(x_distance_to_probe)))[0]  
            values = field[indexes_of_probe]
            y_coords_of_values = y[indexes_of_probe] / delta[indexes_of_probe]
            ax.plot(y_coords_of_values, values, color=color, linewidth=0.8, marker=probe_marker, markevery=3)
     
    handles = []
    for name, probe_marker in zip(names, probe_markers):
        handles.append(Line2D([0], [0], label=name, color="black", linestyle=" ", marker=probe_marker))
    for probe_name, color in zip(probe_names, colors):
        handles.append(Line2D([0], [0], label=probe_name, color=color, linestyle="solid"))
    ax.legend(handles=handles, loc="best", frameon=False)

    if ylim is not None:
        plt.ylim((0, ylim))
    if xlim is not None:
        plt.xlim((0, xlim))

    # plt.legend(bbox_to_anchor=(0.895,-0.50), loc="lower right", 
    #         bbox_transform=fig.transFigure)
    xlabel_mathmode = xlabel.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
    ylabel_mathmode = ylabel.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
    plt.xlabel(xlabel_mathmode)
    plt.ylabel(ylabel_mathmode)
    title_mathmode_modified = title_mathmode.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") # "axial velocities\n__MATHMODESTART__u_{{ref}} = max(u_{{x,LES}}) = {ref_velocity:.3f} m/s__MATHMODEEND__"
    # plt.grid(zorder=-100)
    plt.title(title_mathmode_modified)
    plt.savefig(f"{export_path}{name_template}.png", dpi=300, bbox_inches="tight")

    # plt.title("")
    # plt.savefig(f"{export_path}{name_template}.pdf", dpi=300, bbox_inches="tight")

    print(f"plotted: {export_path}{name_template}.png")

def plot_multiprobe_line_plot(probe_coordinates : list, coordinates : list, fields : list, names : list, 
                xlabel : str, ylabel : str, title_mathmode : str, export_path : str, name_template : str,
                value_scaling : float, ylim : float):

    print("plotting multiprobe_line_plot")
    colors = ["blue", "red", "green", "cyan", "purple"]
    colors[len(coordinates)-1] = "black"

    fig = plt.figure(figsize=(4,2))

    #plt.rcParams['font.size'] = '16'

    for probe_coordinate in probe_coordinates:
    
        for coordinate, field, name, i in zip(coordinates[::-1], fields[::-1], names[::-1], range(len(names))):
            
            x_anchor = probe_coordinate
            x = coordinate[:, 0]
            y = coordinate[:, 1]

            x_distance_to_probe = np.abs(x-probe_coordinate)

            indexes_of_probe = np.where((x_distance_to_probe == np.min(x_distance_to_probe)))[0]            
            values = field[indexes_of_probe] * value_scaling
            y_coords_of_values = y[indexes_of_probe]

            # # area under line
            # inylim = np.where(y_coords_of_values <= ylim)
            # area = np.trapz(values[inylim],y_coords_of_values[inylim])

            plt.scatter(values+x_anchor, y_coords_of_values, s=1, marker="o", color=colors[i])
            plt.plot(values+x_anchor, y_coords_of_values, color=colors[i], linewidth=0.8)
            # plt.text(1.1*np.mean(values)+x_anchor, 0.9*ylim-0.3*(i/ylim), f"{area:.4f}", c=colors[i])
            # print(f"{area:.4f}")
        
    
    plt.ylim((0, np.min([ylim, 1.1*np.max(y_coords_of_values)])))

    lg = plt.legend(names[::-1], bbox_to_anchor=(0.895,-0.56), loc="lower right", 
            bbox_transform=fig.transFigure)

    xlabele_mathmode = xlabel.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") 
    plt.xlabel(xlabele_mathmode)
    plt.ylabel(ylabel)
    
    title_mathmode_modified = title_mathmode.replace("__MATHMODESTART__", "${").replace("__MATHMODEEND__", "}$") # "axial velocities\n__MATHMODESTART__u_{{ref}} = max(u_{{x,LES}}) = {ref_velocity:.3f} m/s__MATHMODEEND__"
    plt.grid(zorder=-100)
    plt.title(title_mathmode_modified)
    plt.savefig(f"{export_path}{name_template}.png", dpi=300, bbox_inches="tight")

    plt.title("")
    plt.savefig(f"{export_path}{name_template}.pdf", dpi=300, bbox_inches="tight")
    
    # paper like version
    # fig.set_figheight(2)
    # fig.set_figwidth(5)
    # lg.remove()
    # plt.legend(names[::-1], fancybox=False, framealpha=1.0)
    # plt.savefig(f"{export_path}{name_template}_small.pdf", dpi=300, bbox_inches="tight")

    print(f"plotted: {export_path}{name_template}.png")



def save_figure_in_multiple_formats(current_figure, fname, size=None):
    """
    saves figure
    """
    try:
        print(f"changing mlp settings")
        # mlp.use("pgf") # mlp.use("agg")
        # mlp.rcParams.update({
        #     "pgf.texsystem": "pdflatex",
        #     'font.family': 'serif',
        #     'text.usetex': True,
        #     'pgf.rcfonts': False,
        #     'mathtext.fontset' : 'dejavuserif'
        # })
        current_figure.savefig(f"{fname}.png", dpi=300, bbox_inches='tight')
        print(f"saved figure {fname}.png")
        if size is not None:
            current_figure.set_figheight(size[0])
            current_figure.set_figwidth(size[1])
        current_figure.savefig(f"{fname}.pdf", dpi=300, bbox_inches='tight')
        print(f"saved figure {fname}.pdf")
        #current_figure.savefig(f"{fname}.pgf", bbox_inches="tight")
        #print(f"saved figure {fname}.pgf")
    except:
        print("failed saving figures with latex style. now changing backend")
        # import matplotlib as mlp
        # from matplotlib import pyplot as plt
        plt.switch_backend("agg")
        mlp.use("agg", force=True) # mlp.use("agg")
        mlp.rcParams.update({
            "pgf.texsystem": "pdflatex",
            #'font.family': 'serif',
            'text.usetex': False,
            #'pgf.rcfonts': False,
            #'mathtext.fontset' : 'dejavuserif'
        })      
        try:
            current_figure.savefig(f"{fname}_agg.png", dpi=300, bbox_inches='tight')
            print(f"saved figure {fname}_agg.png")
            current_figure.savefig(f"{fname}_agg.pdf", dpi=300, bbox_inches='tight')
            print(f"saved figure {fname}_agg.pdf")
        except:
            print("failed saving figure with agg backend")
    
def plot_stencil_vectors(states_as_array, export_path, manual_clim=[-1,1], cmap="nipy_spectral", clabel=None):
    """
    """
    fig, ax = plt.subplots(figsize=(7,4))      
    im = ax.imshow(states_as_array.T, interpolation="none", cmap=cmap)
    ax.set_yticks([0,100,200,300])
    ax.set_aspect(0.7*states_as_array.shape[0]/states_as_array.shape[1])
    im.set_clim(manual_clim[0], manual_clim[1])
    for y in [50,150,250]:
        ax.annotate('', xy=(0, y), xytext=(-800, y), 
            fontsize=10, ha='right', va='center',
            arrowprops=dict(arrowstyle='-[, widthB=3.5, lengthB=0.3', lw=0.5))
    ax.set_xlabel(r"cell index $i$")
    ax.set_ylabel("stencil vector index")
    text_x_pos = -1000
    ax.text(text_x_pos, 0,r"$\hat{r}_{s}$", ha="center", va='center')
    ax.text(text_x_pos, 50,r"$\hat{u}_{x}$", ha="center", va='center')
    ax.text(text_x_pos, 150,r"$\hat{u}_{r}$", ha="center", va='center')
    ax.text(text_x_pos, 250,r"$\hat{\nu}_{t}$", ha="center", va='center')
    ax.yaxis.set_label_coords(-0.12,0.5)
    # fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.04, 0.7])
    if clabel is not None:
        fig.colorbar(im, cax=cbar_ax, label=fr"{clabel}")
    else:
        fig.colorbar(im, cax=cbar_ax)
    plt.savefig(export_path, bbox_inches="tight", dpi=200)
    plt.close()

