# COVID-19 Lung Infection Detection

## Overview
This project leverages machine learning techniques, particularly Convolutional Neural Networks (CNNs) and transfer learning with a pre-trained VGG model, to detect and segment COVID-19-related lung infections from CT scans. The developed model achieves an impressive accuracy of **92.20%**, demonstrating its potential as an aid for healthcare professionals.

---

## Abstract
The study applies advanced medical image analysis to address challenges posed by the COVID-19 pandemic. By integrating effective data handling, model design, and configuration, the framework offers a reliable tool for automated CT image analysis, streamlining diagnostic workflows for COVID-19.

---

## Project Scope
The primary aim is to:
- Distinguish between COVID-19 and Non-COVID regions in CT scan images.
- Employ a pre-trained VGG model for transfer learning to improve performance and efficiency.

Key features:
- Use of the SARS-COV-2 CT-Scan Dataset.
- Preprocessing, augmentation, and segmentation techniques.
- Binary classification (COVID vs. Non-COVID).

---

## Methodology

### 1. Data Handling
- Utilized the **SARS-COV-2 CT-Scan Dataset**, comprising labeled CT scans.
- Split dataset into **training**, **validation**, and **testing** sets.
- Employed `ImageDataGenerator` for on-the-fly data augmentation and normalization.

### 2. Model Architecture
#### CNN Model
- Sequential architecture with:
  - Convolutional layers using ReLU activation.
  - MaxPooling layers for dimensionality reduction.
  - Dropout layers to reduce overfitting.
  - Fully connected Dense layers.

#### VGG Transfer Learning
- Utilized pre-trained VGG16 model.
- Fine-tuned for COVID-19 segmentation using:
  - Flattening and Dense layers for classification.
  - Transfer learning to leverage features learned on the ImageNet dataset.

### 3. Training and Configuration
- Configurations:
  - Input shape: `(224, 224, 3)` for VGG model.
  - Color mode: RGB.
  - Loss function: Categorical Crossentropy.
  - Optimizer: Adam.
  - Batch size: 32.
- Early stopping and model checkpoints ensure optimal training without overfitting.

---

## Results
- **Training Accuracy**: 87.63%.
- **Testing Accuracy**: 92.20%.
- Key outputs include accurate segmentation of infected and non-infected lung regions from CT scans.

### Example Outputs:
1. Correctly segmented COVID-19 regions in CT scans.
2. Identified non-COVID regions with minimal false positives.

---

## Future Development
1. **Dataset Expansion**:
   - Incorporate larger, more diverse datasets for improved generalization.
2. **Architectural Enhancements**:
   - Experiment with different CNN architectures.
   - Integrate additional pre-processing techniques.
3. **Clinical Integration**:
   - Collaborate with healthcare providers for real-time diagnostic applications.
4. **Multimodal Approaches**:
   - Combine CT scan data with clinical metadata for enhanced accuracy.
5. **Ethical Considerations**:
   - Ensure interpretability and fairness in model predictions.

---

## Installation and Usage

### Requirements
- Python 3.x
- TensorFlow/Keras
- Pandas
- NumPy
- scikit-learn
- OpenCV
- Matplotlib

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/covid-lung-detection.git
   cd covid-lung-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Preprocess data:
   ```python
   from preprocessing import split_data, data_generators
   train_gen, val_gen, test_gen = data_generators(data_dir, img_size, batch_size, class_mode, color_mode)
   ```
4. Train the model:
   ```python
   from model import Classifier
   model = Classifier.VGG_model(INPUT_SHAPE)
   history = model.fit(train_gen, validation_data=val_gen, epochs=25, callbacks=[checkpoint, early_stopping])
   ```
5. Test the model:
   ```python
   loss, acc = model.evaluate(test_gen)
   print(f"Test Accuracy: {acc * 100:.2f}%")
   ```

---

## Contributors
- **Hafsa Hafeez Siddiqui**

---

## Acknowledgements
- **Miss Zahida Naz**, Instructor, Machine Learning Lab (AIL-302), Bahria University Karachi Campus.

