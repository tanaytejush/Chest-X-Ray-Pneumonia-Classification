# Chest X-Ray Pneumonia Classification

Binary classification of chest X-ray images as **Normal** or **Pneumonia**, using a logistic regression classifier written from scratch in NumPy (no deep learning framework).

## Approach

- Images are loaded, resized to **64×64** grayscale and flattened into feature vectors.
- Features are normalised using the training set statistics.
- A `ChestXRayClassifier` class implements logistic regression with mini-batch gradient descent, plus prediction, probabilities and evaluation.
- The notebook plots the class distribution, sample images, training curves and example predictions.

## Results on the test set (1,172 images)

| Metric | Score |
|--------|-------|
| Accuracy | 92.7% |
| Precision | 0.948 |
| Recall | 0.952 |
| F1 | 0.950 |

Of the 1,172 test images, 1,086 were classified correctly: 45 normal scans were flagged as pneumonia (false positives) and 41 pneumonia scans were missed (false negatives).

## Run it

The notebook expects the chest X-ray dataset in a local folder. A commonly used source is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset on Kaggle. Set the `data_dir` variable in the notebook to the dataset location.

```bash
pip install numpy pandas matplotlib pillow jupyter
jupyter notebook "Chest X-Ray Pneumonia Classification.ipynb"
```

## Tech stack

Python, NumPy, pandas, matplotlib, Jupyter

## Disclaimer

This is an educational project. It is not a medical device and must not be used for diagnosis.


---

<sub>Made by [tanaytejush](https://github.com/tanaytejush)</sub>
