# Deep Learning Model for Early Alzheimer's Prediction

A simple, reproducible Colab/Notebook project that trains a convolutional model to classify brain MRI slices into three categories:
- AD — Alzheimer's Disease
- CI — Cognitive Impairment
- CN — Cognitively Normal

The core work is implemented in `Alzheimers_Detection.ipynb`.

## Project Overview

This repository contains a Colab-ready Jupyter notebook that:
- Loads axial MRI slices from a directory structure,
- Converts images to grayscale,
- Applies data augmentation to balance classes,
- Normalizes images and splits the dataset into train/validation/test sets,
- Prepares data for training a deep learning classifier (TensorFlow / Keras).

The notebook is oriented toward experimentation and reproducibility rather than production deployment.

## Notebook

File: `Alzheimers_Detection.ipynb`  
Path (notebook used in examples): `/tmp/Axial` (the notebook loads data from this path in the example cells)

Open the notebook in Colab or Jupyter to run the full pipeline and train the model.

## Quick Start

1. Clone the repository:
   git clone https://github.com/YuvaChaithanya/Deep-Learning-Model-For-Early-Alzheimers-Prediction.git
   cd Deep-Learning-Model-For-Early-Alzheimers-Prediction

2. Open `Alzheimers_Detection.ipynb` in Google Colab or Jupyter Notebook.

3. If using Colab:
   - Mount Google Drive (notebook contains `drive.mount('/content/drive')`).
   - Upload or place your dataset archive (e.g., `data.zip`) into Colab and extract as shown in the notebook, or provide a path to the unzipped folder containing class subfolders.

4. Ensure the dataset directory structure is:
   /<dataset_root>/
     AD/...
     CI/...
     CN/...

5. Run the notebook cells sequentially.

## Data

- The notebook expects image files organized by class in subdirectories (Keras `image_dataset_from_directory` format).
- Classes detected in the example: `['AD', 'CI', 'CN']`.
- Images are resized to 128×128 and converted to grayscale in the pipeline.
- The notebook contains a simple augmentation pipeline to balance classes via random flips.

Note: This repository does not include patient-identifiable data. Ensure you have proper permissions to use any medical images.

## Requirements

Minimum recommended environment:
- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- Jupyter or Google Colab

Install common packages with:
pip install -r requirements.txt
(You can create a requirements file from the packages listed above.)

## Key Implementation Notes

- Data augmentation: random flips (horizontal and vertical) are applied to expand minority classes.
- Images are normalized using min-max scaling (division by 255.0).
- Stratified train/validation/test splits are used to preserve class balance.
- The notebook demonstrates converting RGB images to grayscale before training.

## Results

The notebook includes visualization of samples and class distributions. Training and final evaluation metrics are generated within the notebook during experimentation. Check the training cells for accuracy, loss curves, and confusion matrix outputs.

## Contributing

Contributions are welcome. If you'd like to:
- Improve the model architecture,
- Add detailed evaluation metrics and plots,
- Add a training script or saved model export,

open an issue or submit a pull request with a clear description of changes.

## License

This project is provided "as-is" for research and educational purposes. Add a LICENSE file if you would like to apply a specific license.

## Contact

Repository owner: YuvaChaithanya  
For questions or collaboration, please open an issue or contact via GitHub.
