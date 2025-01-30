import numpy as np
import PyFoam as pf
import os
import shutil
from setup_utils import PROJECT_ROOT

def format_float(value):
    """
    Custom float formatter to handle specific formatting requirements.
    """
    # Threshold for using scientific notation
    sci_notation_threshold = 1e-04

    if abs(value) < sci_notation_threshold:
        if abs(value) < 1e-15:  # Consider very small values as zero
            return "0"
        else:
            formatted = f"{value:.5e}"
            # Remove '+' from the exponent part
            formatted = formatted.replace("e+", "e")
            return formatted
    else:
        # Use regular floating-point format
        return f"{value:.9f}"


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

def inject_array(foamfile, ar):
    with open(foamfile, 'r') as file:
        lines = file.readlines()

    start_idx = None
    # Find where 'internalField' starts and ends
    for i, line in enumerate(lines):
        if 'internalField' in line:
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("internalField not found in the file")

    end_idx = lines.index(')\n', start_idx)

    # Open the file again to rewrite it with custom formatting
    with open(foamfile, 'w') as file:
        # Write the parts before 'internalField'
        file.writelines(lines[:start_idx + 3])  # Including header and size declaration

        # Write the 'internalField' with custom formatting
        for row in ar:
            formatted_row = ' '.join(format_float(comp) for comp in row)
            if ar.shape[-1] == 1:
                file.write(f"{formatted_row}\n")
            else:
                file.write(f"({formatted_row})\n")

        # Write the parts after 'internalField'
        file.writelines(lines[end_idx:])

def inject_by_files(numpy_file, template_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    new_file_name = os.path.splitext(numpy_file)[0] + '_foam'
    shutil.copy(template_file, new_file_name)
    input_ar = np.load(numpy_file+'.npy')
    inject_array(new_file_name, foam_format(input_ar))


def inject_by_dirs(numpy_files_dir, template_dir=None, output_dir=None):
    # Ensure injected numpy files directory exists

    if template_dir is None:
        template_dir = numpy_files_dir
    if output_dir is None:
        output_dir = numpy_files_dir
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over numpy files
    for file in os.listdir(numpy_files_dir):
        if file.endswith('.npy'):
            # Path to the original numpy file
            original_file_path = os.path.join(numpy_files_dir, file)

            # New file name without '.npy' extension
            new_file_name = os.path.splitext(file)[0]

            # Path to the new file in injected numpy files directory
            new_file_path = os.path.join(output_dir, new_file_name)
            # Copy template file to new location
            shutil.copy(template_dir, new_file_path)

            # Call your modify_file function
            input_ar = np.load(os.path.join(numpy_files_dir, file))
            inject_array(new_file_path, foam_format(np.load(original_file_path)))


if __name__ == '__main__':



    # Paths to directories
    numpy_files_dir = os.path.join(os.environ['PROJECT_ROOT'], 'saved_data', 'newdata', 'np')
    template_dir = os.path.join(os.environ['PROJECT_ROOT'], 'saved_data', 'newdata', 'foam_template')
    injected_numpy_files_dir = os.path.join(os.environ['PROJECT_ROOT'], 'saved_data','newdata', 'foam_from_np')

    # Ensure injected numpy files directory exists
    os.makedirs(injected_numpy_files_dir, exist_ok=True)

    # Iterate over numpy files
    for file in os.listdir(numpy_files_dir):
        if file.endswith('.npy'):
            # Path to the original numpy file
            original_file_path = os.path.join(numpy_files_dir, file)

            # New file name without '.npy' extension
            new_file_name = os.path.splitext(file)[0]

            # Path to the new file in injected numpy files directory
            new_file_path = os.path.join(injected_numpy_files_dir, new_file_name)
            template_path = os.path.join(template_dir, 'template')
            # Copy template file to new location
            shutil.copy(os.path.join(template_dir, 'template'), new_file_path)

            # Call your modify_file function
            input_ar = np.load(os.path.join(numpy_files_dir, file))
            inject_array(new_file_path, foam_format(np.load(original_file_path)))
