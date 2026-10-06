# Convolution and Transfer Learning with VGG16

A practical neural-network assignment exploring **2-D convolution from scratch** and **transfer learning with VGG16** through frozen feature extraction and fine-tuning.

The notebook first implements convolution manually with NumPy to show how a filter moves across an input matrix and produces a feature map. It then compares two transfer-learning approaches using the same cat-and-dog image dataset: one where the pretrained VGG16 convolutional/base layers remain frozen, and another where the final convolutional block is unfrozen and fine-tuned.

## What This Project Covers

### 1. Implement Convolution from Scratch

The first task implements a **2-D convolution/filtering operation without using a built-in convolution function**.

The program:

- stores the input matrix and filter as NumPy arrays;
- slides the `3 × 3` filter across the `5 × 5` input matrix;
- uses stride `1` and padding `0`;
- computes the dot product at every valid location;
- prints the complete output feature map; and
- prints the output shape.

The completed convolution produced the following output:

![Convolution output feature map and shape](images/Convolution%20result.png)

The final output feature map was:

```text
[[4 3 4]
 [2 4 3]
 [2 3 4]]
```

with an output shape of:

```text
(3, 3)
```

If the stride is changed from `1` to `2`, the filter moves two positions at a time instead of one. This reduces the number of valid filter positions and changes the output size from `3 × 3` to `2 × 2`.

---

### 2. Transfer Learning: Freeze vs. Fine-Tune

The second task compares two transfer-learning approaches using a pretrained **VGG16** convolutional neural network.

The same setup is used for both experiments so the results can be compared fairly:

- pretrained VGG16 with ImageNet weights;
- CIFAR-10 dataset;
- only the **cat** and **dog** classes;
- Cat = `0`;
- Dog = `1`;
- images resized to `96 × 96`;
- batch size = `32`; and
- training for `5` epochs.

The shared setup prepares the dataset, image preprocessing, and training settings once so they do not need to be repeated for both experiments.

---

## Experiment A — Feature Extraction

In Experiment A, the pretrained VGG16 network is used as a **frozen feature extractor**.

The convolutional/base layers are frozen so that their pretrained ImageNet weights do not change during training. The original VGG16 classification layers are removed and replaced with:

- `GlobalAveragePooling2D`; and
- a `Dense(1)` output layer with Sigmoid activation for binary cat-and-dog classification.

Because the VGG16 base is frozen, only the new classifier is trained.

### Model Parameters

The model summary shows that only **513 parameters** are trainable, while the pretrained VGG16 convolutional parameters remain frozen.

![Experiment A model summary](images/Experiment%20A-%20%20Model%20summary.png)

| Parameter Type | Count |
| --- | ---: |
| Total Parameters | 14,715,201 |
| Trainable Parameters | 513 |
| Non-Trainable Parameters | 14,714,688 |

The 513 trainable parameters come from the final classifier:

- `512` weights from the pooled VGG16 features; and
- `1` bias.

### Training Results

Experiment A was trained for **5 epochs**.

The training output shows that accuracy improved across the epochs while the loss decreased.

![Experiment A training output](images/Experiment%20A%20-%20Training.png)

The main results were:

| Metric | Result |
| --- | ---: |
| Trainable Parameters | 513 |
| Training Time | 823.42 seconds |
| Final Training Accuracy | 84.61% |
| Test Accuracy | 83.0% |
| Final Training Loss | 0.3526 |

The training loss decreased from approximately `1.0933` in the first epoch to `0.3526` in the fifth epoch.

![Experiment A training loss](images/Experiment%20A%20-%20%20Training%20loss%20graph.png)

This shows that the new classifier was able to learn from the visual features already extracted by the frozen VGG16 network.

---

## Experiment B — Fine-Tuning

Experiment B also starts with a pretrained VGG16 network, but this time the final convolutional block is allowed to learn.

The earlier VGG16 blocks remain frozen while **block5** is unfrozen. This allows the final convolutional block and the new classifier to update their weights during training.

### Trainable Layers

The layer output confirms that blocks 1–4 remained frozen while the `block5` layers were trainable.

![Experiment B trainable layers](images/Experiment%20B%20-%20%20%20trainable%20layers.png)

This satisfies the fine-tuning requirement of unfreezing at least the last convolutional block.

### Model Parameters

Unfreezing block5 greatly increased the number of trainable parameters.

![Experiment B model summary](images/Experiment%20B%20-%20Model%20summary.png)

| Parameter Type | Count |
| --- | ---: |
| Total Parameters | 14,715,201 |
| Trainable Parameters | 7,079,937 |
| Non-Trainable Parameters | 7,635,264 |

Compared with the 513 trainable parameters in Experiment A, the fine-tuned model trained more than seven million parameters.

### Training Results

Experiment B was also trained for the same **5 epochs** so that the comparison with Experiment A remained fair.

![Experiment B training output](images/Experiment%20B%20-%20Training.png)

The main results were:

| Metric | Result |
| --- | ---: |
| Trainable Parameters | 7,079,937 |
| Training Time | 974.44 seconds |
| Final Training Accuracy | 98.87% |
| Test Accuracy | 84.85% |
| Final Training Loss | 0.0589 |

The training loss decreased from approximately `0.6707` in the first epoch to `0.0589` in the fifth epoch.

![Experiment B training loss](images/Experiment%20B%20-%20Training%20loss%20graph.png)

The fine-tuned network learned the training data much more strongly than the frozen feature extractor. However, the difference between the final training accuracy of **98.87%** and the test accuracy of **84.85%** suggests some overfitting.

---

## Feature Extraction vs. Fine-Tuning

The two experiments show the trade-off between keeping a pretrained network frozen and allowing part of it to adapt to the new dataset.

| Method | Trainable Parameters | Training Time | Accuracy |
| --- | ---: | ---: | ---: |
| Frozen Feature Extractor | 513 | 823.42 seconds | 83.0% |
| Fine-Tuned Network | 7,079,937 | 974.44 seconds | 84.85% |

The notebook comparison table is shown below:

![Feature extraction and fine-tuning comparison](images/Model%20Comparism%20Table%20.png)

The **Frozen Feature Extractor** required far fewer trainable parameters because the pretrained VGG16 convolutional/base layers were kept unchanged. It trained faster and still achieved a test accuracy of **83.0%**.

The **Fine-Tuned Network** trained the last VGG16 convolutional block together with the classifier. This increased the trainable parameter count to **7,079,937** and increased the training time to **974.44 seconds**. Fine-tuning improved test accuracy to **84.85%**, but the improvement was relatively small compared with the large increase in trainable parameters and computation.

The fine-tuned network also reached a much higher training accuracy than test accuracy, indicating that it fit the training data very closely and showed some overfitting.

Overall, the comparison demonstrates that **feature extraction is computationally cheaper**, while **fine-tuning gives the model more flexibility to adapt pretrained features to the new dataset** and can improve predictive performance.

---

## Running the Notebook

The project uses Python in Jupyter Notebook/JupyterLab.

### Required Libraries

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook or JupyterLab

If the required packages are not already installed, they can be installed from a terminal or command prompt:

```bash
pip install tensorflow numpy pandas matplotlib jupyter
```

Or directly from a Jupyter Notebook cell:

```python
%pip install tensorflow numpy pandas matplotlib
```

Restart the Jupyter kernel after installation if required.

Then open the assignment notebook and run the cells from top to bottom.

### Dataset

The CIFAR-10 dataset is loaded with:

```python
tf.keras.datasets.cifar10.load_data()
```

If CIFAR-10 is not already cached locally, TensorFlow downloads it the first time this cell is run.

Only the cat and dog classes are retained for this assignment.

### Pretrained VGG16 Weights

The pretrained model is loaded with:

```python
VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
```

If the ImageNet weights are not already cached locally, TensorFlow downloads them the first time VGG16 is loaded.

### Training Environment

Training time depends on the available hardware and TensorFlow configuration. In this run, TensorFlow displayed a warning that native Windows TensorFlow versions above 2.10 do not provide standard CUDA GPU support, so the recorded times reflect the environment used for this notebook.

---

## Key Takeaways

This assignment demonstrates several important convolutional neural network and transfer-learning concepts:

- A 2-D convolution can be implemented manually by sliding a filter across an input matrix and computing the dot product at every valid location.
- Filter size, stride, and padding determine the size of the resulting feature map.
- Increasing stride reduces the number of filter positions and therefore reduces the output spatial size.
- Transfer learning allows knowledge learned from a large dataset such as ImageNet to be reused for another image-classification problem.
- Freezing pretrained convolutional layers keeps their learned weights unchanged and greatly reduces the number of trainable parameters.
- Fine-tuning allows selected pretrained layers to adapt to a new dataset, but it increases computation and training time.
- The frozen feature extractor achieved **83.0%** test accuracy with only **513 trainable parameters**.
- Fine-tuning improved test accuracy to **84.85%**, but increased the trainable parameter count to **7,079,937**.
- The large gap between fine-tuning training accuracy and test accuracy suggests some overfitting.
- Comparing both methods demonstrates the practical trade-off between computational efficiency and model adaptability.

---

## Repository Structure

```text
Neural Networks_HA3/
│
├── Neural Networks_HA3.ipynb
├── README.md
│
└── images/
    ├── Convolution result.png
    ├── Experiment A-  Model summary.png
    ├── Experiment A - Training.png
    ├── Experiment A -  Training loss graph.png
    ├── Experiment B -   trainable layers.png
    ├── Experiment B - Model summary.png
    ├── Experiment B - Training.png
    ├── Experiment B - Training loss graph.png
    └── Model Comparism Table .png
```

---

## Technologies Used

**Python · TensorFlow · Keras · NumPy · Pandas · Matplotlib · VGG16 · CIFAR-10 · Jupyter Notebook**
