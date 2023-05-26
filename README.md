Fruit Classifier using ResNet-18
This repository contains code for training a fruit classifier using the ResNet-18 architecture in PyTorch. The classifier is trained on a dataset of fruits, which is divided into training and validation sets.

Dataset
The fruit dataset used in this project should be structured as follows:

lua
Copy code
/kaggle/input/60-fruit-types-curated-dataset
    |-- A - E
    |       |-- Fruit1
    |       |       |-- image1.jpg
    |       |       |-- image2.jpg
    |       |       |-- ...
    |       |
    |       |-- Fruit2
    |       |-- ...
    |
    |-- F-M
    |
    |-- M-R
    |
    |-- S-Y
Please ensure that your dataset follows this structure. If you have a different directory or folder structure, you will need to modify the root_dir and subfolders variables accordingly.

Installation
To run this code, you need to have Python 3 and the following libraries installed:

torch
torchvision
scikit-learn
tqdm
matplotlib
You can install the required libraries by running the following command:

Copy code
pip install torch torchvision scikit-learn tqdm matplotlib
Training
To train the fruit classifier, follow these steps:

Clone this repository: git clone https://github.com/your-username/fruit-classifier.git
Change into the project directory: cd fruit-classifier
Run the training script: python train.py
During training, the script will print the loss and accuracy for both the training and validation sets after each epoch. It will also save the model to a ".pth" file every 5 epochs.

Results
After training, the script will plot a graph showing the training and validation loss as a function of the epochs. Additionally, it will display the index and value of the lowest validation loss.

GPU Support
By default, the code is set to use GPU if available. If you don't have a GPU or want to use CPU instead, modify the following line in train.py:

python
Copy code
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Change "cuda:0" to "cpu".

Note: Running the code on a GPU can significantly speed up the training process.

License
This project is licensed under the MIT License. See the LICENSE file for more information.

Feel free to fork and modify this repository to suit your needs. If you have any questions or suggestions, please feel free to open an issue.

Happy fruit classification!
