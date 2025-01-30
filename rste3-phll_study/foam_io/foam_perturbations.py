import numpy as np
import PyFoam as pf
import subprocess
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
case_path = '/CFD_dataset/openfoam/test/case_1p2'




field_names = ['epsilon',
               'f',
               'k',
               'nut',
               'p',
               'phi',
               'phit',
               'U']





def format_float(value):
    """
    Custom float formatter to handle specific formatting requirements.
    """
    # Threshold for using scientific notation
    sci_notation_threshold = 1e-05

    if abs(value) < sci_notation_threshold:
        if abs(value) < 1e-15:  # Consider very small values as zero
            return "0"
        else:
            formatted = f"{value:.6e}"
            # Remove '+' from the exponent part
            formatted = formatted.replace("e+", "e")
            return formatted
    else:
        # Use regular floating-point format
        return f"{value:.6f}"


class FoamCase:
    def __init__(self, path, cur_itr=0):
        self.path = path
        self.current_iteration = cur_itr
        self.system_path = os.path.join(self.path, 'system')
        self.control_dict_path = os.path.join(self.path, 'system', 'controlDict')

    def modify_time_controls(self, start=None, end=None):
        with open(self.control_dict_path, 'r') as file:
            lines = file.readlines()

        with open(self.control_dict_path, 'w') as file:
            for line in lines:
                # Check if the line starts with 'startTime' or 'endTime'
                if line.startswith('startTime') and start is not None:
                    file.write(f'startTime       {start};\n')
                elif line.startswith('endTime') and end is not None:
                    file.write(f'endTime         {end};\n')
                else:
                    file.write(line)

    def read_velocity_field(self, t=None):
        if t is None:
            case_files = os.listdir(self.path)
            max_t = -1
            for cname in case_files:
                if cname.isnumeric():
                    if int(cname) > max_t:
                        max_t = int(cname)
            t = max_t
        max_t = t
        return ParsedParameterFile(os.path.join(self.path, str(t), 'U'))

    def read_coords(self, t=None):
        if t is None:
            case_files = os.listdir(self.path)
            max_t = -1
            for cname in case_files:
                if cname.isnumeric():
                    if int(cname) > max_t:
                        max_t = int(cname)
            t = max_t
        max_t = t
        return ParsedParameterFile(os.path.join(self.path, str(t), 'C'))

    def noise_velocity_field(self, t=None, magnitude=0.1):
        if t is None:
            case_files = os.listdir(self.path)
            max_t = -1
            for cname in case_files:
                if cname.isnumeric():
                    if int(cname) > max_t:
                        max_t = int(cname)
            t = max_t

        velocity_field_path = os.path.join(self.path, str(t), 'U')
        velocity_field = ParsedParameterFile(velocity_field_path)

        n_cells = len(velocity_field['internalField'])
        noise = magnitude * (np.random.rand(n_cells, 3) - 0.5) * 2
        for i in range(n_cells):
            velocity_field['internalField'][i] *= (1 + noise[i])

        # Read the entire file and store parts other than 'internalField'
        with open(velocity_field_path, 'r') as file:
            lines = file.readlines()

        start_idx = None
        # Find where 'internalField' starts and ends
        for i, line in enumerate(lines):
            if 'internalField' in line and 'nonuniform List<vector>' in line:
                start_idx = i
                break

        if start_idx is None:
            raise ValueError("internalField not found in the file")

        end_idx = lines.index(')\n', start_idx)

        # Open the file again to rewrite it with custom formatting
        with open(velocity_field_path, 'w') as file:
            # Write the parts before 'internalField'
            file.writelines(lines[:start_idx + 3])  # Including header and size declaration

            # Write the 'internalField' with custom formatting
            for vec in velocity_field['internalField']:
                formatted_vec = ' '.join(format_float(comp) for comp in vec)
                file.write(f"({formatted_vec})\n")

            # Write the parts after 'internalField'
            file.writelines(lines[end_idx:])

    def get_velocity_field(self, t=None, visualize=False):
        if t is None:
            case_files = os.listdir(self.path)
            max_t = -1
            for cname in case_files:
                if cname.isnumeric():
                    if int(cname) > max_t:
                        max_t = int(cname)
            t = max_t
        max_t = t
        velocity_field = self.read_velocity_field(t=max_t)
        coords = self.read_coords(t=max_t)

        velocity_array = np.array(velocity_field['internalField'])
        coords_array = np.array(coords['internalField'])
        dirs = ['x', 'y', 'z']
        if visualize:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            for i in range(3):
                sc = axs[i].scatter(coords_array[:, 0], coords_array[:, 1], c=velocity_array[:, i])
                plt.colorbar(sc, ax=axs[i])
                axs[i].set_title(f'U{dirs[i]}')
            plt.show()

        return velocity_array, coords_array

    def run_simulation(self, n_iters=10):
        case_files = os.listdir(self.path)
        max_t = -1
        for cname in case_files:
            if cname.isnumeric():
                if int(cname) > max_t:
                    max_t = int(cname)
        start = max_t
        end = max_t + n_iters
        self.modify_time_controls(start=start, end=end)
        cwd = os.getcwd()
        os.chdir(self.path)
        subprocess.run(['wsl', '/mnt/c/Users/GrzegorzKaszuba/runFoam.sh'])
        os.chdir(cwd)

def visualize_velocity_difference(field1, field2, coords, title='Velocity Difference and Magnitude'):
    """
    Visualize the difference between two velocity fields and the magnitude of the first field.

    :param field1: First velocity field as a NumPy array.
    :param field2: Second velocity field as a NumPy array.
    :param coords: Coordinates as a NumPy array.
    :param title: Title of the plot.
    """
    # Calculate the difference in velocity
    difference = np.linalg.norm(field1 - field2, axis=1)
    magnitude_field1 = np.linalg.norm(field1, axis=1)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot difference in velocity magnitude
    sc1 = axs[0].scatter(coords[:, 0], coords[:, 1], c=difference, cmap='viridis')
    plt.colorbar(sc1, ax=axs[0], label='Difference in velocity magnitude')
    axs[0].set_xlabel('X Coordinate')
    axs[0].set_ylabel('Y Coordinate')
    axs[0].set_title('Velocity Difference')

    # Plot magnitude of the first velocity field
    sc2 = axs[1].scatter(coords[:, 0], coords[:, 1], c=magnitude_field1, cmap='viridis')
    plt.colorbar(sc2, ax=axs[1], label='Magnitude of first velocity field')
    axs[1].set_xlabel('X Coordinate')
    axs[1].set_ylabel('Y Coordinate')
    axs[1].set_title('Magnitude of First Velocity Field')

    plt.suptitle(title)
    plt.show()


if __name__ == "__main__":
    case = FoamCase(case_path, cur_itr=400)
    velocity_field, coords = case.get_velocity_field(visualize=True)
    case.run_simulation(n_iters=5)

    # Save the velocity field after running the simulation
    post_sim_velocity, _ = case.get_velocity_field(visualize=False)

    # Add noise to the velocity field
    case.noise_velocity_field(magnitude=0.1)

    # Save the noisy velocity field
    noisy_velocity, _ = case.get_velocity_field(visualize=True)

    case.run_simulation(n_iters=5)
    noisy_velocity_post, _ = case.get_velocity_field(visualize=True)


    visualize_velocity_difference(velocity_field, post_sim_velocity, coords)
    visualize_velocity_difference(post_sim_velocity, noisy_velocity, coords)
    visualize_velocity_difference(post_sim_velocity, noisy_velocity_post, coords)