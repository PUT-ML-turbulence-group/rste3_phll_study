import numpy as np
import PyFoam as pf
import subprocess
import os
from dataclasses import dataclass
import matplotlib as mpl
import matplotlib.pyplot as plt
from setup_utils import PROJECT_ROOT
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
case_path = 'C:\\Users\\GrzegorzKaszuba\\PycharmProjects\\RSTe3\\CFD_dataset\\openfoam\\kepsilonphitf\\pehill\\case_1p2'

def foam_format(ar):
    if ar.ndim == 1:
        ar = ar.reshape(-1, 1)
    if ar.shape[1:] == (3, 3):
        ar = ar[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    elif ar.shape[1:] == (3,):
        pass
    elif ar.shape[1:] == (1,):
        pass
    elif ar.shape[1:] == (6,):
        pass
    else:
        raise NotImplementedError('No defined way to reshape this kind of array')
    return ar


class CaseVisualization:
    def __init__(self, case_path):
        self.path = case_path
        self.coords = self.read_field('C', visualize=False)
        self.system_path = os.path.join(self.path, 'system')
        self.control_dict_path = os.path.join(self.path, 'system', 'controlDict')

    def read_field(self, field_name='U', t=None, visualize=False, title=''):
        if t is None:
            case_files = os.listdir(self.path)
            max_t = -1
            for cname in case_files:
                if cname.isnumeric():
                    if int(cname) > max_t:
                        max_t = int(cname)
            t = max_t
        field_array = np.array(ParsedParameterFile(os.path.join(self.path, str(t), field_name))['internalField'])
        if visualize:
            coords_array = self.coords
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            for i in range(3):
                sc = axs[i].scatter(coords_array[:, 0], coords_array[:, 1], c=field_array[:, i])
                plt.colorbar(sc, ax=axs[i])
                axs[i].set_title(f'{field_name} {title}')
            plt.show()


        return field_array


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

        return velocity_array, coords_array


    def visualize_scalars_difference(self, field1, field2, title=''):
        """
        Visualize the difference between two velocity and the magnitude of the first field.

        :param field1: First velocity field as a NumPy array.
        :param field2: Second velocity field as a NumPy array.
        :param coords: Coordinates as a NumPy array.
        :param title: Title of the plot.
        """
        field1 = foam_format(field1)
        field2 = foam_format(field2)
        # Calculate the difference in velocity
        difference = field1 - field2
        magnitude_field1 = np.linalg.norm(field1, axis=1)

        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        # Plot difference in velocity magnitude
        sc1 = axs[0].scatter(self.coords[:, 0], self.coords[:, 1], c=difference, cmap='viridis')
        plt.colorbar(sc1, ax=axs[0], label='Difference in velocity magnitude')
        axs[0].set_xlabel('X Coordinate')
        axs[0].set_ylabel('Y Coordinate')
        axs[0].set_title(f'Difference of fields')

        # Plot magnitude of the first velocity field
        sc2 = axs[1].scatter(self.coords[:, 0], self.coords[:, 1], c=magnitude_field1, cmap='viridis')
        plt.colorbar(sc2, ax=axs[1], label='Magnitude of first velocity field')
        axs[1].set_xlabel('X Coordinate')
        axs[1].set_ylabel('Y Coordinate')
        axs[1].set_title(f'Magnitude of first field')

        plt.suptitle(f'Difference visualization')
        plt.show()


    def visualize_tensors_difference(self, field1, field2, title=''):
        """
        Visualize the difference between two velocity and the magnitude of the first field.

        :param field1: First velocity field as a NumPy array.
        :param field2: Second velocity field as a NumPy array.
        :param coords: Coordinates as a NumPy array.
        :param title: Title of the plot.
        """
        field1 = foam_format(field1)
        field2 = foam_format(field2)
        # Calculate the difference in velocity
        difference = field1 - field2
        magnitude_field1 = np.linalg.norm(field1, axis=1)
        magnitude_field2 = np.linalg.norm(field2, axis=1)

        # Create subplots
        fig, axs = plt.subplots(1, 5, figsize=(15, 6))

        sc1 = axs[0].scatter(self.coords[:, 0], self.coords[:, 1], c=difference[:, 0], cmap='viridis')
        plt.colorbar(sc1, ax=axs[0], label='Difference in velocity magnitude')
        axs[0].set_xlabel('X Coordinate')
        axs[0].set_ylabel('Y Coordinate')
        axs[0].set_title('Difference of xx component')


        sc1 = axs[1].scatter(self.coords[:, 0], self.coords[:, 1], c=difference[:, 1], cmap='viridis')
        plt.colorbar(sc1, ax=axs[0], label='Difference in velocity magnitude')
        axs[0].set_xlabel('X Coordinate')
        axs[0].set_ylabel('Y Coordinate')
        axs[0].set_title('Difference of yy component')


        sc1 = axs[2].scatter(self.coords[:, 0], self.coords[:, 1], c=difference[:, 3], cmap='viridis')
        plt.colorbar(sc1, ax=axs[0], label='Difference in velocity magnitude')
        axs[0].set_xlabel('X Coordinate')
        axs[0].set_ylabel('Y Coordinate')
        axs[0].set_title('Difference of xy component')
        # Plot difference in magnitude
        sc1 = axs[3].scatter(self.coords[:, 0], self.coords[:, 1], c=np.linalg.norm(difference, axis=1), cmap='viridis')
        plt.colorbar(sc1, ax=axs[0], label='Magnitude of first field')
        axs[0].set_xlabel('X Coordinate')
        axs[0].set_ylabel('Y Coordinate')
        axs[0].set_title('Magnitude of first field')

        # Plot magnitude of the first field
        sc2 = axs[4].scatter(self.coords[:, 0], self.coords[:, 1], c=magnitude_field1, cmap='viridis')
        plt.colorbar(sc2, ax=axs[1], label='Magnitude of second field')
        axs[1].set_xlabel('X Coordinate')
        axs[1].set_ylabel('Y Coordinate')
        axs[1].set_title('Magnitude of second field')

        plt.suptitle(title)
        plt.show()




case = CaseVisualization(case_path)
rst_rans = case.read_field('turbulencePropertiesR')
rst_dns = np.load(os.path.join(os.environ['PROJECT_ROOT'], 'saved_data', 'numpy', 'rst_horizontal_dns.npy'))
case.visualize_tensors_difference(rst_dns, rst_rans)
