# Project Overview

This project aims to optimize and analyze the processed RHEED data using machine learning methods. This will facilitate the initialization of material growth parameters and enhance the efficient optimization of InAs quantum dot (QD) growth under programmatic control. It includes codes, small dataset to demo the code, and Labview programs tailored for deployment on specific equipment. Below is a detailed description of the folder structure and contents.

## Code Folder
This folder contains the script codes and ONNX files. 
* **00**: This is the main script that will call scripts **01**, **02**, and **03** in sequence to demonstrate the data preprocessing, model training, and model inference process.The **00** scripts are divided into multiple **Section**, each separated by multiple lines of **'*'**:   
   * In the First Section:
      * **script_name_01** matches the name of the **01** script, indicating that this Section will call the **01** script.   
      * `input_paths` and `output_paths` are used to set the read path for preprocessed data and the storage path for preprocessed results.
   * In the Second Section:
      * `base_directory` is used to set the folder path for data restructuring.
   * In the Third Section:
      *  `configs` are used to set parameters for each training loop:
         * `data_path`: Path for the model training data.
         * `num_classes`: Number of classes the model should classify.
         * `csv_path`: Path to store the results generated during model training.
         * `weights_path_template`: Path to store the weight information generated during model training.
      * **script_name_02** matches the name of the **02** script, indicating that this Section will call the **02** script.
   * In the Fourth Section**:
      * `model_paths` and `folder_paths` are used to set the ONNX file path and the data used for inference.
      * **script_name_03** matches the name of the **03** script, indicating that this Section will call the **03** script.
* **01**: This script is responsible for data preprocessing, including image enhancement and converting multiple image stitches to NPY format.
* **02**: This script demonstrates the model training process using the pre-processed results from the script **01**. 
* **03**: This script is to demonstrate the model inference process, which will call the ONNX format models in the **04** folder and apply them to process the PNG format data in the **data** folder.
* **04**: This folder stores the ONNX file, which was converted from the model pre-trained with complete dataset in this research. The **03** script will call these ONNX files to demonstrate the model inference process.

## Data Folder
This folder contains small dataset related to the three models used in this research to demo the code:
- Data is organized by model name and label name for convenient loading and use.

## How to Run the Example Code
 **00** script calls **01**, **02** and **03** scripts in sequence to demonstrate the data preprocessing, model training, and model inference process. To run the **00** script, follow these steps:
1. **Prepare the Data**;
   * Please confirm that the **"Initialization model"**, **"Shutter model"**, and **"Temperature model"** folders all exist in the path **/data/PNG**.
   * Verify the completeness of scripts **01**, **02**, and **03** and the **04** folder under the **code** directory. The **04** folder should contain 3 ONNX files.
2. **Run the **00** script**
   * Select the **00** script with the mouse and right-click to select **Set as File to Run**;
   * Left mouse click on **Reproducible run** in the top right corner of the page.

## Results Folder
This folder contains 3 subfolders and a text file:
1. **CSV:** This folder stores the data generated during the training process of each model.
2. **NPY:** This folder contains the NPY files that have undergone data preprocessing and data restructuring for convenient access during model training.
3. **Weights:** This folder stores the model parameters generated during the training of each model.
4. **output**: This file contains the final results which show as a record of the log information generated during the model training process.

## Labview program Folder
This folder contains the Labview program designed for the Riber 32P system:
- Temperature reading and control are implemented using the universal Eurotherm serial communication protocol.
- The shutter controller code is written based on the system manual.
- The camera interface uses USB 3.0 for data acquisition.
- The program is deployed in this research and supports real-time RHEED data processing and feedback control.
- Before running the program, ensure the following folders are manually created and correctly set:
  - **Real time storage Excel Folder**: For real-time data output.
  - **Image save Folder**: For storing RHEED image data.
  - **ONNX File**: Path for checking and calling ONNX files.

## Notes
1. Ensure that all required environments and dependencies are properly installed for running the codes and performing ONNX model inference.
2. The Labview program is exclusively designed for the Riber 32P system and requires appropriate configuration to match the specific system environment.
3. Verify all paths and dependencies before running the codes to prevent errors during execution.
4. In order to reduce the amount of space taken up by the results, **--epochs** is set to 10 in script **00** and **select_stride** is set to 20 in script **01**.
5. Please use a compatible browser to open the relevant links and fold the content.

## Contact Information
For further information or questions, please contact the research team.
