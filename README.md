# Mango Leaf Disease Classifier

This project classifies mango leaf diseases using a convolutional neural network and K-Nearest Neighbors (KNN). It includes a GUI application for user interaction.

## Project Structure

- `MLDC.ipynb`: Main Jupyter notebook for training models and evaluating performance.
- `main.py`: GUI application to load and classify images.
- `Dataset/`: Folder containing image data for training and testing.
- `MLDC.keras`: Saved model for feature extraction.
- `MLDCKNNClassifier.joblib` and `MLDCLabelEncoder.joblib`: Trained KNN classifier and label encoder.
- `requirements.txt`: File containing project dependencies.

## Setup Instructions

1. Clone the repository:

    ```bash
    git clone https://github.com/username/repo_name.git
    cd repo_name
    ```

2. Install required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Place your dataset in the `Dataset` folder with subfolders for each class of disease.

4. Run `MLDC.ipynb` to train the model and generate required files, if not already provided.

5. Run the GUI application:

    ```bash
    python main.py
    ```

## Using the GUI

1. Launch the application.
2. Click "Upload Image" to upload an image.
3. Click "Predict Leaf Disease" to see the prediction and accuracy.

## Notes

- The project assumes a folder structure for the dataset, where each class of images is stored in separate folders within the `Dataset` directory.
- If `MLDC.keras`, `MLDCKNNClassifier.joblib`, or `MLDCLabelEncoder.joblib` are missing, run the notebook to regenerate them.
