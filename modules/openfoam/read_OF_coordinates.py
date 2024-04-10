import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt



def read_OF_internal_field_results(filename, path_to_folder):
    """
    reads results for example from u file
    """

    results = []

    coordinate_head = '(\n'
    coordinate_end = ')\n'

    with open(path_to_folder+filename) as f:
        lines = f.readlines()

        internalField_start_lines = [no for no, line in enumerate(
            lines) if "internalField" in line]
        internalField_start_line = internalField_start_lines[0]
        coordinate_head_lines = [no for no, line in enumerate(
            lines) if coordinate_head in line]
        coordinate_end_lines = [no for no, line in enumerate(
            lines) if coordinate_end == line]
        if (len(coordinate_head_lines) > 0) and (len(coordinate_end_lines) > 0):
            coord_start_idx = [
                idx for idx in coordinate_head_lines if idx > internalField_start_line][0]
            coord_end_idx = [
                idx for idx in coordinate_end_lines if idx > coord_start_idx][0]
            values = []
            for line in lines[coord_start_idx+1:coord_end_idx]:
                values.append(evaluate_OF_vector_line(line))
        else:
            values = None

    # convert list of lists to numpy array
    results = np.array(values)

    return results

def evaluate_OF_vector_line(vector_line : str) -> list:
    """
    this function converts the string '(5.1172 -0.00180226 -0.00426538)\n'
    to a list [5.1172,-0.00180226,-0.00426538]
    """
    try:
        replace_chars = ["(", ")", "\n"]
        for ch in replace_chars:
            vector_line = vector_line.replace(ch, "")
        vector_entry_as_list = vector_line.split(" ")
        vector_entry_as_list = [float(elem) for elem in vector_entry_as_list]
    except:
        vector_entry_as_list = None
        
    return vector_entry_as_list

def read_OF_internal_field_coordinates(path_to_folder):
    """
    the command 'writeCellCentres'creates ccx, ccy, ccz files
    this function extracts the x,y,z points of a specified boundary
    """

    results = []

    filenames = ["ccx", "ccy", "ccz"]

    boundary_name = "inlet"
    coordinate_head = '(\n'
    coordinate_end = ')\n'

    for filename in filenames:

        with open(path_to_folder+filename) as f:
            lines = f.readlines()

            internalField_start_lines = [no for no, line in enumerate(
                lines) if "internalField" in line]
            internalField_start_line = internalField_start_lines[0]
            coordinate_head_lines = [no for no, line in enumerate(
                lines) if coordinate_head in line]
            coordinate_end_lines = [no for no, line in enumerate(
                lines) if coordinate_end in line]
            coord_start_idx = [
                idx for idx in coordinate_head_lines if idx > internalField_start_line][0]
            coord_end_idx = [
                idx for idx in coordinate_end_lines if idx > coord_start_idx][0]
            values = []
            for line in lines[coord_start_idx+1:coord_end_idx]:
                values.append(eval(line))

            results.append(values)

    # convert list of lists to numpy array
    results = np.array(results).T

    return results




def read_OF_inlet_coordinates(path_to_folder="/home/lukas/Data/OF-Tests/simpleFoamCase/templates/"):
    """
    the command 'writeCellCentres'creates ccx, ccy, ccz files
    this function extracts the x,y,z points of a specified boundary
    """

    results = {}

    filenames = ["ccx", "ccy", "ccz"]

    boundary_name = "inlet"
    coordinate_head = '(\n'
    coordinate_end = ')\n'

    for filename in filenames:

        with open(path_to_folder+filename) as f:
            lines = f.readlines()

            boundary_start_lines = [no for no, line in enumerate(
                lines) if boundary_name in line]
            boundary_start_line = boundary_start_lines[0]
            coordinate_head_lines = [no for no, line in enumerate(
                lines) if coordinate_head in line]
            coordinate_end_lines = [no for no, line in enumerate(
                lines) if coordinate_end in line]
            coord_start_idx = [
                idx for idx in coordinate_head_lines if idx > boundary_start_line][0]
            coord_end_idx = [
                idx for idx in coordinate_end_lines if idx > coord_start_idx][0]
            values = []
            for line in lines[coord_start_idx+1:coord_end_idx]:
                values.append(eval(line))

            results[filename] = values

    return results


def write_OF_boundary_condition(sim_inlet_coordinates, inlet_condictions):
    """
    """
    template_folder = "/home/lukas/Data/OF-Tests/simpleFoamCase/templates/"
    target_folder = "/home/lukas/Data/OF-Tests/simpleFoamCase/0/"
    path_to_template_file = "U_template"
    name_of_new_file = "U"

    with open(template_folder+path_to_template_file) as f:
        #lines = f.readlines()
        complete_text = f.read()
        counter = inlet_condictions[0].shape[0]
        vectors_text = ""
        for i in range(counter):
            vectors_text += f"(   {inlet_condictions[0][i]:.4f}   {inlet_condictions[1][i]:.4f}  {inlet_condictions[2][i]:.4f})\n"

        complete_text = complete_text.replace("#counter", str(counter))
        complete_text = complete_text.replace("#vectors", vectors_text)

    with open(target_folder+name_of_new_file, 'w') as f:
        f.write(complete_text)

def interpolate_inlet_condition_data(x, y, ux, ur, ut, indexes_of_inlet, sim_inlet_coordinates, export_path):
    """
    This function interpolates between the inlet condition positions between the mean field data
    and the required coordinates for the openFoam Mesh
    """

    ux_by_y_interp = interpolate.interp1d(
        y[indexes_of_inlet], ux[indexes_of_inlet])
    ur_by_y_interp = interpolate.interp1d(
        y[indexes_of_inlet], ur[indexes_of_inlet])
    ut_by_y_interp = interpolate.interp1d(
        y[indexes_of_inlet], ut[indexes_of_inlet])

    inlet_condition_ux = ux_by_y_interp(sim_inlet_coordinates["ccy"])
    inlet_condition_ur = ur_by_y_interp(sim_inlet_coordinates["ccy"])
    inlet_condition_ut = ut_by_y_interp(sim_inlet_coordinates["ccy"])

    inlet_conditions = [inlet_condition_ux,
                        inlet_condition_ur, inlet_condition_ut]

    datasets = [ux, ur, ut]
    labels = ["ux", "ur", "ut"]
    inlet_conditions = [inlet_condition_ux, inlet_condition_ur, inlet_condition_ut]
    for label, data, inlet_condition in zip(labels, datasets, inlet_conditions):
        fig = plt.figure()
        plt.scatter(sim_inlet_coordinates["ccy"], inlet_condition,
                    s=1.5, color="red", label="inlet condition interpolation")
        plt.scatter(y[indexes_of_inlet], data[indexes_of_inlet],
                    s=1.5, color="blue", label="mean field data")
        plt.xlabel("y")
        plt.ylabel(label)
        plt.title(f"interpolated data at inlet boundary")
        plt.grid()
        plt.legend()
        plt.savefig(f"{export_path}inlet_condition_{label}.png", dpi=300)


        fig = plt.figure()
        plt.scatter(sim_inlet_coordinates["ccx"],
                    sim_inlet_coordinates["ccy"], s=1, color="red")
        plt.scatter(x[indexes_of_inlet], y[indexes_of_inlet], s=1, color="blue")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"coordinates of inlet boundary")
        plt.savefig(f"{export_path}coordinates of inlet boundary.png", dpi=300)
        plt.close()

    return inlet_conditions

if __name__ == "__main__":

    results = read_OF_coordinates()
