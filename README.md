# Machine Learning-driven Alignment Architecture of Heterogeneous Data with Transient Varying Semantics

## Instructions
The software and code involved in paper "Machine Learning-driven Alignment Architecture of Heterogeneous Data with the Transient Varying Semantics", including the synchronize triggering software and code for data generation, data processing and data alignment, as well as obtaining arc detection models.

## System Requirements

### Hardware Requirements
The operating system used in this study was Windows 10, supported by:
- Intel Core i5
- NVIDIA GeForce 3090 hardware  
- 48 GB of RAM

### Software Requirements
- Python 3.9.6
- PyTorch 2.2.2+cu118
- Scikit-learn 1.5.1
- Numpy 1.23.5
- Pandas 2.2.2
- Matplotlib 3.8.4
- Regex 2.5.148
- Opencv-python 4.6.0
- Pillow 10.3.0
- Torchvision 0.17.2+cpu
- Xlrd 2.0.1
- Xlwt 1.3.0

### Software Installation Instructions
Use the pip command in the terminal window to configure the above software environment, and contact the established environment in Sublime Text.

## Demo
Here, demonstration cases corresponding to Figures 4. d-f and Supplementary Figs. 31a-b in the paper are provided.

**Running method:** Download the script from the Demo folder and run it in Sublime Text.

**Expected output:** The maximum test accuracy or minimum mean square error will reach its maximum or minimum value at a sample shift of 0.

**Run time:** Run time can be almost ignored and forgotten, you can try to reproduce it.

## Reproducibility
The script used for reproduction is stored in the "Task reproduction" folder. The index of the corresponding task table is shown in "Task Reproduction Index Table.pdf". The dataset used for alignment is stored in [Dataset](URL).

The file directory for storing source code is the same as the folder for the corresponding dataset. Download the script and corresponding dataset from the repository, place the script and corresponding "data" folder in the same root directory, and run the script using "Sublime Text" to export the corresponding time shift/sample shift-test accuracy/mean square error/maximum test accuracy/minimum mean square error curve or time shift-rank correlation coefficient curve to Excel file.

The reference range for running these scripts ranges from less than 1 second to several tens of hours.
