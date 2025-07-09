"""**************************************************************************************************************"""
"""**************************************************************************************************************"""
"""**************************************************************************************************************"""

import subprocess

# Define the Python file name to run
script_name_01 = "01 Data preprocessing to convert PNG format to NPY file.py"

# Define different input and output paths
input_paths = ["/data/PNG/Reconstruction model", 
               "/data/PNG/Shutter model", 
               "/data/PNG/Temperature model"
              ]
output_paths = ["/results/NPY/Reconstruction model", 
                "/results/NPY/Shutter model", 
                "/results/NPY/Temperature model"
               ]

# Traverse each pair of input and output paths and run the script
for input_path, output_path in zip(input_paths, output_paths):
    # Construct the command to run
    command = [
        "python",
        script_name_01,
        input_path,
        output_path
    ]

    # Run the command and capture the output
    result = subprocess.run(command, text=True, capture_output=True)

    # Print output results
    print(f"Running with input: {input_path} and output: {output_path}")
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)


"""**************************************************************************************************************"""
"""**************************************************************************************************************"""
"""**************************************************************************************************************"""


import os
import shutil

def delete_csv_files(base_dir):
    
    print(f"Start checking and deleting all. csv files：{base_dir}")
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):  # Check if the file ends in. csv
                source_path = os.path.join(root, file)  # Build the complete path of the file
                print(f"Delete file {source_path}")
                os.remove(source_path)  # Delete. csv file


def move_npy_files(base_dir):
    
    # Use the os-walk function to recursively traverse the directory tree, with topdown=False indicating traversal from subdirectories upwards
    print(f"Start traversing the directory：{base_dir}")
    for root, dirs, files in os.walk(base_dir, topdown=False):
        print(f"\nCurrent directory：{root}")
        print(f"The included files：{files}")
        print(f"Subdirectories included：{dirs}")
        
        for file in files:
            if file.endswith('.npy'):  # Check if the file ends with. npy
                source_path = os.path.join(root, file)  # Build the complete path of the file
                destination_path = os.path.join(os.path.dirname(root), file)  # Build target path (upper level directory)
                print(f"move file from {source_path} to {destination_path}")
                shutil.move(source_path, destination_path)  # Move the. npy file to the target path

        # Traverse the list of subdirectories in the current directory
        for dir in dirs:
            dir_path = os.path.join(root, dir)  # Build the complete path of the subdirectories
            if not os.listdir(dir_path):  # Check if the subdirectories are empty
                print(f"remove empty directories：{dir_path}")
                os.rmdir(dir_path)  # Delete empty subdirectories
                
def rename_folders(base_dir):
    
    print(f"Start checking and modifying folder names：{base_dir}")
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if "select_width_size" in dir_name:  # Check if the directory name contains 'select_width_size'
                old_dir_path = os.path.join(root, dir_name)
                # Extract the part before the first space as the new directory name
                new_dir_name = dir_name.split(" select_width_size")[0]
                new_dir_path = os.path.join(root, new_dir_name)
                
                # rename directory
                print(f"Change folder name: {old_dir_path} -> {new_dir_path}")
                os.rename(old_dir_path, new_dir_path)
                
# Call the function to specify base-directory as the path to the folder that needs to be organized
base_directory = '/results/NPY/'  # Replace with the actual root directory path

delete_csv_files(base_directory)
move_npy_files(base_directory)
rename_folders(base_directory)


"""**************************************************************************************************************"""
"""**************************************************************************************************************"""
"""**************************************************************************************************************"""

import subprocess

# OSError: Cannot save file into a non-existent directory: '/results/CSV'
if not os.path.exists("/results/CSV"):
        os.makedirs("/results/CSV")


# Parameter configuration
configs = [
    {
        "data_path": "/results/NPY/Reconstruction model",
        "num_classes": 6,
        "csv_path": "/results/CSV/Reconstruction_Model_Acc_and_Loss.csv",
        "weights_path_template": "/results/Weights/weights_Reconstruction_Model/"
    },
    {
        "data_path": "/results/NPY/Shutter model",
        "num_classes": 2,
        "csv_path": "/results/CSV/Shutter_Model_Acc_and_Loss.csv",
        "weights_path_template": "/results/Weights//weights_Shutter_Model/"
    },
    {
        "data_path": "/results/NPY/Temperature model",
        "num_classes": 3,
        "csv_path": "/results/CSV/Temperature_Model_Acc_and_Loss.csv",
        "weights_path_template": "/results/Weights//weights_Temperature_Model/"
    }
]

# Script file path
script_name_02 = "02 (Train loop)select_width_size 24 width_length.py"

# Run each configuration
for config in configs:
    command = [
        "python", script_name_02,
        "--data_path", config["data_path"],
        "--num_classes", str(config["num_classes"]),
        "--csv_path", config["csv_path"],
        "--weights_path_template", config["weights_path_template"],
        "--batch_size", "128",
        "--epochs", "10",
        "--lr", "0.01",
        "--device", "cuda:0"
    ]
    
    print(f"Running command: {' '.join(command)}")
    
    # Use Popen to read and display output line by line
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Real time output standard output and error output
    for stdout_line in process.stdout:
        print(stdout_line, end='')  # Output standard output

    for stderr_line in process.stderr:
        print(stderr_line, end='')  # Output error output
    
    process.wait()  # Waiting for the child process to end


"""**************************************************************************************************************"""
"""**************************************************************************************************************"""
"""**************************************************************************************************************"""


import subprocess
import sys

# Define different model paths and folder paths
model_paths = [
    "/code/04 ONNX file/Reconstruction_model.onnx",
    "/code/04 ONNX file/Reconstruction_model.onnx",
    "/code/04 ONNX file/Reconstruction_model.onnx",
    "/code/04 ONNX file/Reconstruction_model.onnx",
    "/code/04 ONNX file/Reconstruction_model.onnx",
    "/code/04 ONNX file/Reconstruction_model.onnx",
    "/code/04 ONNX file/Reconstruction_model.onnx",
    "/code/04 ONNX file/Reconstruction_model.onnx",
    "/code/04 ONNX file/Reconstruction_model.onnx",
    "/code/04 ONNX file/Reconstruction_model.onnx",
    "/code/04 ONNX file/Reconstruction_model.onnx",
    "/code/04 ONNX file/Reconstruction_model.onnx",

    "/code/04 ONNX file/Shutter_model.onnx",
    "/code/04 ONNX file/Shutter_model.onnx",
    "/code/04 ONNX file/Shutter_model.onnx",
    "/code/04 ONNX file/Shutter_model.onnx",

    "/code/04 ONNX file/Temperature_model.onnx",
    "/code/04 ONNX file/Temperature_model.onnx",
    "/code/04 ONNX file/Temperature_model.onnx",
    "/code/04 ONNX file/Temperature_model.onnx",
    "/code/04 ONNX file/Temperature_model.onnx",
    "/code/04 ONNX file/Temperature_model.onnx"
]

folder_paths = [
    "/data/PNG/Reconstruction model/0 As cap/1",
    "/data/PNG/Reconstruction model/0 As cap/2",
    "/data/PNG/Reconstruction model/1 c(4×4)/1",
    "/data/PNG/Reconstruction model/1 c(4×4)/2",
    "/data/PNG/Reconstruction model/2 (2×4)/1",
    "/data/PNG/Reconstruction model/2 (2×4)/2",
    "/data/PNG/Reconstruction model/3 Oxidation/1",
    "/data/PNG/Reconstruction model/3 Oxidation/2",
    "/data/PNG/Reconstruction model/4 Deoxidation/1",
    "/data/PNG/Reconstruction model/4 Deoxidation/2",
    "/data/PNG/Reconstruction model/5 (n×6)/1",
    "/data/PNG/Reconstruction model/5 (n×6)/2",


    "/data/PNG/Shutter model/0 No/1",
    "/data/PNG/Shutter model/0 No/2",
    "/data/PNG/Shutter model/1 Yes/1",
    "/data/PNG/Shutter model/1 Yes/2",

    "/data/PNG/Temperature model/0 High/1",
    "/data/PNG/Temperature model/0 High/2",
    "/data/PNG/Temperature model/1 Low/1",
    "/data/PNG/Temperature model/1 Low/2",
    "/data/PNG/Temperature model/2 Suitable/1",
    "/data/PNG/Temperature model/2 Suitable/2"
    
]

# Ensure that the target script path is correct
script_name_03 = "03 Read small batches of data and model inference.py"

# Traverse the list of model paths and folder paths
for onnx_model_path, folder_path in zip(model_paths, folder_paths):
    # Run the target script and pass the model path and folder path as parameters
    process = subprocess.run(
        [sys.executable, script_name_03, onnx_model_path, folder_path],
        capture_output=True,
        text=True
    )

    # Standard output and error output of the output script
    print(process.stdout)
    print(process.stderr)



"""**************************************************************************************************************"""
"""**************************************************************************************************************"""
"""**************************************************************************************************************"""



