import matplotlib.pyplot as plt
import numpy as np






def extract_inlet_velocity(x_coordinate_of_inlet, x, rho, theta, ux, ur, ut, export_path):
    """
    extract indexes of points on inlet boundary and plot the data on the inlet boundary
    """

    # -----------------------------------------
    # extract inlet velocity

    distance_to_inlet = np.abs(x-x_coordinate_of_inlet)
    indexes_of_inlet = np.where( 
        (distance_to_inlet == np.min(distance_to_inlet)))

    #indexes_of_inlet = np.where( 
    #    (np.abs(x-x_coordinate_of_inlet)<4*1e-4))# & 


    rho_inlet = rho[indexes_of_inlet]
    theta_inlet = theta[indexes_of_inlet]
    x_inlet = x[indexes_of_inlet]
    ux_inlet = ux[indexes_of_inlet]
    ur_inlet = ur[indexes_of_inlet]
    ut_inlet = ut[indexes_of_inlet]

    size = 0.1

    print("now plotting inlet velocities")
    fig = plt.figure()
    plt.scatter(ur_inlet, rho_inlet, s=size)
    plt.xlabel("ur")
    plt.ylabel("rho")
    plt.title(f"ur over rho at inlet (at x={x_coordinate_of_inlet})")
    plt.savefig(f"{export_path}ur over rho at inlet.png", dpi=300)

    fig = plt.figure()
    plt.scatter(ux_inlet, rho_inlet, s=size)
    plt.xlabel("ur")
    plt.ylabel("rho")
    plt.title(f"ux over rho at inlet (at x={x_coordinate_of_inlet})")
    plt.savefig(f"{export_path}ux over rho at inlet.png", dpi=300)

    fig = plt.figure()
    plt.scatter(ut_inlet, rho_inlet, s=size)
    plt.xlabel("ut")
    plt.ylabel("rho")
    plt.title(f"ut over rho at inlet (at x={x_coordinate_of_inlet})")
    plt.savefig(f"{export_path}ut over rho at inlet.png", dpi=300)

    fig = plt.figure()
    plt.scatter(x_inlet, rho_inlet, s=size)
    plt.xlabel("x")
    plt.ylabel("rho")
    plt.title(f"x over rho at inlet (at x={x_coordinate_of_inlet})")
    plt.savefig(f"{export_path}x over rho at inlet.png", dpi=300)

    fig = plt.figure()
    plt.scatter(x_inlet, theta_inlet, s=size)
    plt.xlabel("x")
    plt.ylabel("theta")
    plt.title(f"x over theta at inlet (at x={x_coordinate_of_inlet})")
    plt.savefig(f"{export_path}x over theta at inlet.png", dpi=300)

    # -----------------------------------------

    return indexes_of_inlet