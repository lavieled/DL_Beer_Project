# <h1 align="center">ECE 046217 - Technion - Deep Learning - Project </h1> 
## <h2 align="center"> "Thats not what i ordered!" - A Deep Learning beer classification model </h2>

<h4 align="center">
  <table align="center" style="border: none;">
    <tr style="border: none;">
      <td align="center" style="border: none;">
        <div>
          <img src="./data/readme/lavie_drink.jpg" width="100" height="100"/> <br>
          <strong>Lavie Lederman</strong> <br>
          <a href="https://www.linkedin.com/in/lavie-lederman/">
            <img src="./data/readme/LinkedInLogo.png" width="40" height="40"/>
          </a>
          <a href="https://github.com/lavieled">
            <img src="./data/readme/GitHubLogo.png" width="40" height="40"/>
          </a>
        </div>
      </td>
      <td align="center" style="border: none;">
        <div>
          <img src="./data/readme/barry_think.jpg" width="100" height="100"/> <br>
          <strong>Barry Shafran</strong> <br>
          <a href="https://www.linkedin.com/in/barry-shafran-562979244/">
            <img src="./data/readme/LinkedInLogo.png" width="40" height="40"/>
          </a>
          <a href="https://github.com/barryshaf">
            <img src="./data/readme/GitHubLogo.png" width="40" height="40"/>
          </a>
        </div>
      </td>
    </tr>
    <tr style="border: none;">
      <td colspan="2" align="center" style="border: none">
        <a href="https://youtu.be/..." target="_blank">//presentation vid
          <img src="./data/readme/YouTubeLogo.png" width="50" height="50"/>
        </a>
      </td>
    </tr>
  </table>
</h4>

## Abstract
This project addresses the challenge of visually identifying beer types from real world images, a task complicated by noisy backgrounds, varied presentation styles, and overlapping visual features between classes. We built a custom dataset by scraping images from the Untappd platform and preprocessing them with YOLOv8 to isolate relevant objects. Several state-of-the-art models, including DINO and EfficientNet, were evaluated under different training strategies: feature extraction, partial fine-tuning, and full fine-tuning, alongside a lightweight CNN developed as a simple and low cost baseline. Performance was compared against a “human benchmark” obtained via a custom GUI classification game. Results show that while our lightweight model exceeded human accuracy, transfer learning with partial fine-tuning achieved the best balance between accuracy and computational efficiency, with EfficientNet full fine-tuning reaching the highest overall performance.



## Table of Contents
- [Abstract](#abstract)  
- [Files in the Repository](#files-in-the-repository)  
- [Installation Instructions](#installation-instructions)  
  - [Core Deep Learning & Computer Vision](#core-deep-learning--computer-vision)  
  - [Data Processing & Analysis](#data-processing--analysis)  
  - [Image Processing & Visualization](#image-processing--visualization)  
  - [Web Scraping & Data Collection](#web-scraping--data-collection)  
  - [User Interface & Utilities](#user-interface--utilities)  
- [How to Use](#how-to-use)  
- [Gathering the Dataset](#gathering-the-dataset)  
- [Imported Models](#imported-models)  
  - [Hyperparameter Tuning](#hyperparameter-tuning)  
- [Training Results](#training-results)  
  - [Our Model’s Architecture](#our-models-architecture)  
  - [Model Results with No Data Augmentations](#model-results-with-no-data-augmentations)  
- [Data Augmentations](#data-augmentations)  
- [Results](#results)  
  - [Basic CNN Model Results](#Basic-CNN-model-results)  
  - [DINO Results](#dino-results)  
  - [EfficientNet Results](#efficientnet-results)  
- [References](#references)  


## Files in the repository

| File Name                              | Purpose                                                                                      |
|----------------------------------------|----------------------------------------------------------------------------------------------|
| `login_and_get_links.py`               | Logs in to Untappd and retrieves links for specified beer type photos.                       |
| `download.py`                          | Uses the image links to download all desired photos.                                         |
| `sortToCupNoCup.py`                    | Uses YOLOv8 to identify and crop beer glasses from the photos.                               |
| `selfTest.py`                          | Simple GUI game to test human classification ability, used as a benchmark.                   |
| `fullModelTest.py`                     | Pipelines the process from uncropped, untagged image input to results with bounding boxes and tags. |
| `BeerDL_50_augmented.ipynb`            | Trains a simple CNN beer classifier with data augmentation.                                  |
| `BeerDLNoAugmentations40Epochs.ipynb`  | CNN beer classifier training without augmentations for 40 epochs.                            |
| `BeerDLNoAugmentations50Epochs.ipynb`  | CNN beer classifier training without augmentations for 50 epochs.                            |
| `EffnetAndDinoTraining.ipynb`          | Comparison of EfficientNet and DINO training strategies.                                     |
| `data`                                 | Directory containing datasets and results.                                                   |
| `HW`                                   | Directory containing homework assignments.                                                   |

## Installation Instructions

1. Get Anaconda with Python 3, follow the instructions according to your OS (Windows/Mac/Linux) at [link](https://www.anaconda.com/download)
2. Install the packages bellow.

### Core Deep Learning & Computer Vision

| Library | Installation Command |
|---------|----------------------|
| [PyTorch](https://pytorch.org/) | `pip install torch` |
| [torchvision](https://pytorch.org/vision/stable/) | `pip install torchvision` |
| [opencv-python](https://opencv.org/) | `pip install opencv-python` |
| [ultralytics](https://docs.ultralytics.com/) | `pip install ultralytics` |
| [timm](https://github.com/huggingface/pytorch-image-models) | `pip install timm` |

### Data Processing & Analysis

| Library | Installation Command |
|---------|----------------------|
| [numpy](https://numpy.org/) | `pip install numpy` |
| [pandas](https://pandas.pydata.org/) | `pip install pandas` |

### Image Processing & Visualization

| Library | Installation Command |
|---------|----------------------|
| [Pillow (PIL)](https://pillow.readthedocs.io/) | `pip install Pillow` |
| [matplotlib](https://matplotlib.org/) | `pip install matplotlib` |

### Web Scraping & Data Collection

| Library | Installation Command |
|---------|----------------------|
| [selenium](https://selenium-python.readthedocs.io/) | `pip install selenium` |
| [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/) | `pip install beautifulsoup4` |
| [requests](https://requests.readthedocs.io/) | `pip install requests` |

### User Interface & Utilities

| Library | Installation Command |
|---------|----------------------|
| [tkinter](https://docs.python.org/3/library/tkinter.html) | `pip install tk` |
| [tqdm](https://tqdm.github.io/) | `pip install tqdm` |


## How to Use
After installing the required libraries, you can run the `fullModelTest.py`to see the results for a custom photo.
If you wish to train the models and evaluate them your self you can run the notebooks under `src` which contain the training and evaluation process for the all the models we analysed.
If you wish to test yourself, you can run 'selfTest.py' and have fun. 

You can also download our weights from the following link:
- [Drive](https://technionmail-my.sharepoint.com/:f:/g/personal/barryshafran_campus_technion_ac_il/El2xVVe9De1FprcCWzj5MyEBwzbr2RncuJIcf4dOZvM6gA?e=eUL8cz)
  
  Just download the `models.zip` file and replace it with the existing `./data/models` folder in the repository.

## Gathering the Dataset
We could not find any tagged dataset that would help our project so we used avaliable resources and some deep learning models to create the needed dataset. 
We wrote a [script](./data_changes/login_and_get_links.py) that gathers links for photos from the online website for beer lovers [Untappd]([untapped link](https://untappd.com/)), using the tags from the website itself as out correct labels(for example, we send the script to download all photos of a beer we know is a wheat beer, thus we created a "labeled" dataset, after gathering the links we used another [script](./data_changes/download.py) to download them/
Next, since all the photos we user uploaded(social media like) we had a lot of irrelevant objects in the photos, so we used the pre-trained YOLO_v2 for its object detection and bbox features to get only the images of the beers themselves.
Note that the YOLO model only had options for detecting cups or bottles, so we still had to manuelly delete a few dozen photos, but we used them for out control group as "not beer" label.
This is a small data set(for tests and example), for the full one please access:


 - [Drive](https://technionmail-my.sharepoint.com/:f:/g/personal/barryshafran_campus_technion_ac_il/EtEsbGmXuvFBpTJWI7XmMEoBj0Qxyf32OU3HE59scnnt_Q?e=a2lzY0)

   examples from the data with labels:
   
| <img src="./photos/cider/ace-cider-the-california-cider-company-ace-pineapple-cider_ace-cider-the-california-cider-company-ace-pineapple-cider_002.jpg_crop_0.jpg" width="120"/> | <img src="./photos/ipa/bells-brewery-two-hearted-ipa_bells-brewery-two-hearted-ipa_029.jpg_crop_1.jpg" width="120"/> | <img src="./photos/lager/stella-artois-stella-artois_stella-artois-stella-artois_062.jpg_crop_3.jpg" width="120"/> | <img src="./photos/not_beer_cup/angry-orchard-cider-company-crisp-apple_angry-orchard-cider-company-crisp-apple_023.jpg_crop_0.jpg" width="120"/> | <img src="./photos/stout/founders-brewing-co-breakfast-stout_founders-brewing-co-breakfast-stout_089.jpg_crop_1.jpg" width="120"/> | <img src="./photos/wheat/bells-brewery-oberon-ale_bells-brewery-oberon-ale_025.jpg_crop_0.jpg" width="120"/> |
|---|---|---|---|---|---|
| **Cider** | **IPA** | **Lager** | **Not Beer** | **Stout** | **Wheat** |

## Imported models
In this project we trained (except) the following models and used:

- `efficientnet_b0` [EfficientNet (TensorFlow Implementation)](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) – Used as a benchmark model for comparison.
- `vit_base_patch16_224_dino` [DINO (Facebook Research)](https://github.com/facebookresearch/dino) – Used as a benchmark model for comparison.
- `yolov8n` [YOLOv8 models documentation](https://docs.ultralytics.com/models/yolov8/)

### Hyperparameter Tuning
A comprehensive search was performed across multiple hyperparameter spaces, including model preweight strategies, learning rates, momentum, and weight decay parameters.
The study involved carefully selecting combinations of hyperparameters to achieve the best validation score.
Below is a detailed table of the tuned hyperparameters.

| **Category**   | **Hyperparameter** | **Value / Range Used**               | **Description**                                   |
|----------------|--------------------|---------------------------------------|---------------------------------------------------|
| Model          | `preweight_mode`   | `random`                              | Model trained from scratch (no pretrained weights)|
| Training       | `batch_size`       | `64`                                  | Number of samples per training batch              |
| Training       | `epochs`           | `5` (search), `50` (our CNN) , '20' (improted models)           | Trials for tuning, then final full run            |
| Optimizer      | `optimizer`        | `AdamW`                               | Optimization algorithm                            |
| AdamW          | `lr`               | `[1e-4, 1e-3]` (log scale)            | Learning rate for AdamW (tuned)                   |
| AdamW          | `weight_decay`     | `[1e-6, 1e-3]` (log scale)            | Weight decay regularization (tuned)               |
| Regularization | `dropout`          | `[0.1, 0.6]`                          | Dropout probability in classifier layers (tuned)  |
| Scheduler      | `scheduler`        | `ReduceLROnPlateau`                   | Learning rate scheduler type                      |
| Plateau        | `factor`           | `0.5`                                 | Decay factor for ReduceLROnPlateau                 |
| Plateau        | `patience`         | `[2, 4]`                              | Patience for ReduceLROnPlateau (tuned)             |

- The best configurations for each model was saved for future training.
- At the end, each configuration was set to be trained with a batch size of 64 for 40 epochs.

## Training Results

### our model's architecture

```mermaid
flowchart TD
    A[Input 3×224×224]
    B[Conv 3→64 + BN + ReLU + MaxPool]
    C[Conv 64→128 + BN + ReLU + MaxPool]
    D[Conv 128→256 + BN + ReLU + MaxPool]
    E[Conv 256→512 + BN + ReLU]
    F[AdaptiveAvgPool 1×1]
    G[Flatten]
    H[Dropout p=0.2]
    I[Linear 512→6 · logits]

    A --> B;
    B --> C;
    C --> D;
    D --> E;
    E --> F;
    F --> G;
    G --> H;
    H --> I;

```

### Model results with no data augmentations
<img src="./data/readme/barry_no_aug_results.png" width="400"/> 


*No Augmentations Test Accuracy: **72.30%** — Trained without data augmentations. Accuracy improves rapidly in the first 15 epochs and then plateaus around ~68–70%. The steady loss decrease indicates consistent learning, but the absence of augmentations likely limits the model’s ability to generalize to unseen variations.*

## Data Augmentations

We applied different transformations for training and evaluation to ensure robust learning while keeping validation/testing consistent and unbiased.  
All transforms were implemented using [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html).

### **Training Transformations**
| **Augmentation**          | **Parameters**                                      | **Description**                                                                 |
|---------------------------|------------------------------------------------------|---------------------------------------------------------------------------------|
| `Resize`                  | `(256, 256)`                                        | Resize the input image to a fixed size of 256×256 pixels.                       |
| `RandomCrop`              | `IMG_SIZE`                                          | Randomly crop the image to `IMG_SIZE`×`IMG_SIZE` pixels.                        |
| `RandomHorizontalFlip`    | *(default p=0.5)*                                   | Flip the image horizontally with a probability of 0.5.                          |
| `ColorJitter`             | `(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)` | Randomly change image brightness, contrast, saturation, and hue.                |
| `RandomRotation`          | `15`                                                | Rotate the image randomly within ±15 degrees.                                   |
| `ToTensor`                | —                                                   | Convert the image to a PyTorch tensor.                                          |
| `Normalize`               | `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` | Normalize pixel values using ImageNet statistics.                               |

### **Evaluation Transformations**  
| **Augmentation**          | **Parameters**                                      | **Description**                                                                 |
|---------------------------|------------------------------------------------------|---------------------------------------------------------------------------------|
| `Resize`                  | `(IMG_SIZE, IMG_SIZE)`                              | Resize the input image to `IMG_SIZE`×`IMG_SIZE` pixels.                         |
| `ToTensor`                | —                                                   | Convert the image to a PyTorch tensor.                                          |
| `Normalize`               | `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` | Normalize pixel values using ImageNet statistics.                               |

### **General Transformations**  
| **Augmentation**          | **Parameters**                                      | **Description**                                                                 |
|---------------------------|------------------------------------------------------|---------------------------------------------------------------------------------|
| `Resize`                  | `(224, 224)`                                        | Resize the input image to 224×224 pixels.                                       |
| `ToTensor`                | —                                                   | Convert the image to a PyTorch tensor.                                          |


### **Additional Augmentation Pipeline for Dataset Expansion**

To further increase dataset diversity and size, we created an **additional augmented sample for every original image** using the following transformations:

| **Augmentation**          | **Parameters**                                      | **Description**                                                                 |
|---------------------------|------------------------------------------------------|---------------------------------------------------------------------------------|
| `Resize`                  | `(256, 256)`                                        | Resize the input image to 256×256 pixels.                                       |
| `RandomCrop`              | `224`                                               | Randomly crop the image to 224×224 pixels.                                      |
| `RandomHorizontalFlip`    | *(default p=0.5)*                                   | Flip the image horizontally with a probability of 0.5.                          |
| `RandomRotation`          | `20`                                                | Rotate the image randomly within ±20 degrees.                                   |
| `RandomAffine`            | `degrees=15`, `translate=(0.1, 0.1)`, `scale=(0.9, 1.1)` | Apply random affine transformations with rotation, translation, and scaling.   |
| `RandomPerspective`       | `distortion_scale=0.3`, *(default p=0.5)*           | Apply a random perspective transformation to simulate viewpoint changes.        |
| `ToTensor`                | —                                                   | Convert the image to a PyTorch tensor.                                          |
| `ToPILImage`              | —                                                   | Convert the tensor back to a PIL image (for further processing or visualization).|

**Note:** This process effectively **doubled the dataset size** by generating one additional augmented image for every original photo.

## Results

### Basic CNN model results

<img src="./data/readme/barry_model_results.png" width="400"/> 

*basic CNN model Test Accuracy: **72.59%** — Accuracy improves steadily throughout training, reaching a plateau near the end. Loss decreases consistently, indicating stable learning without major overfitting. However, the final accuracy is lower compared to other approaches, suggesting that this configuration may lack sufficient capacity to capture all dataset-specific patterns.*


### Effnet results

### Model Performance Comparisons

| **DINO – Feature Extraction** | **DINO – Partial Freezing** | **DINO – Full Fine-Tuning** |
|-------------------------------|-----------------------------|-----------------------------|
| ![DINO Feature Extraction](./data/readme/dino_feature_ext_result.png) | ![DINO Partial](./data/readme/dino_partial_result.png) | ![DINO Full](./data/readme/dino_full_result.png) |
| *Final Test Accuracy: **73.91%** — Training is stable but accuracy plateaus early, suggesting that frozen pre-trained features limit adaptation to beer-specific patterns.* | *Final Test Accuracy: **83.53%** — Significant improvement from feature extraction. Accuracy stabilizes after ~5 epochs, showing that fine-tuning later layers captures more domain-specific details.* | *Final Test Accuracy: **80.76%** — Lower than partial freezing despite full fine-tuning. Likely due to overfitting or disruption of pre-trained features during training.* |


### Dino results


| **EfficientNet – Feature Extraction** | **EfficientNet – Partial Freezing** | **EfficientNet – Full Fine-Tuning** |
|---------------------------------------|--------------------------------------|--------------------------------------|
| ![EfficientNet Feature Extraction](./data/readme/effnet_feature_ext_result.png) | ![EfficientNet Partial](./data/readme/effnet_partial_results.png) | ![EfficientNet Full](./data/readme/effnet_full_results.png) |
| *Final Test Accuracy: **67.76%** — The lowest-performing setup overall. Limited adaptation to the dataset due to frozen features.* | *Final Test Accuracy: **83.67%** — Large jump in accuracy, showing partial fine-tuning is highly effective for EfficientNet on this dataset.* | *Final Test Accuracy: **87.78%** — The best performance overall, with a steady accuracy climb and minimal overfitting.* |

## Concluding pipeline example

To conclude our project, we ran a single example image set through **all of our final models** to compare their predictions side-by-side.  
This gives us a clear visual understanding of how each model responds to the same input, making it easy to spot strengths and weaknesses.

**Input photo:**

<p align="center">
  <img src="./data/readme/test2.jpg" width="400"/>
</p>

**All the models output:**

<p align="center">
  <img src="./data/readme/test2_all_models.jpg" width="800"/>
</p>

## References
1. [Untappd – Discover Beer](https://untappd.com) – Source for beer images and type labels.  
2. [Ultralytics – YOLOv8 Models Documentation](https://docs.ultralytics.com/models/yolov8/) – Object detection model used for preprocessing and dataset creation.  
3. [DINO – Emerging Properties in Self-Supervised Vision Transformers (Facebook AI Research)](https://github.com/facebookresearch/dino) – Vision Transformer model used for feature extraction and fine-tuning.  
4. [EfficientNet – TensorFlow Implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) – CNN model used for classification experiments.  
5. [PyTorch](https://pytorch.org/) – Main deep learning framework for training and evaluation.  
6. [TorchVision](https://pytorch.org/vision/stable/) – Image datasets, transforms, and pretrained models.  
7. [timm – PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) – Additional pretrained models and utilities.  
8. [OpenCV](https://opencv.org/) – Computer vision and image processing library.  
9. [Pillow (PIL)](https://pillow.readthedocs.io/) – Image loading and processing library.  
10. [Selenium](https://selenium.dev/documentation/) – Web automation framework used for scraping Untappd.  
11. [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) – HTML/XML parsing library used for scraping.  

