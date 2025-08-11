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
Dont you hate it when you drink a beer and dont know what type is it? or when the waiter mixed up your order?
In this project we tested different state of the art classification models to try to recognize beer types from visual input, and comparing the results we got with different approaches.
using a basic model we built ourselves and a GUI to play a game of human beer detection as a benchmark.
we also used the YOLO object detection model to aide us in gathering data for this project.
<div align="center">
  <img src="./data/readme/yolo_video.gif" alt="Pothole Detection" width="600">//change
</div>

## Table of Contents
* [Files in the repository](#Files-in-the-repository)
* [Installation Instructions](#Installation-Instructions)
  * [Libraries to Install](#Libraries-to-Install)
* [How to Use](#How-to-Use)
* [Dataset](#Dataset)
* [Object Detection Models](#Object-Detection-Models)
  * [Hyperparameter Tuning](#Hyperparameter-Tuning)
* [Training Results](#Training-Results)
* [Data Augmentations](#Data-Augmentations)
* [Post Augmentations Results](#Post-Augmentations-Results)
* [Potholes Severity](#Potholes-Severity)
* [References](#References)

## Files in the repository

| File Name                       | Purpose                                                                 |
|---------------------------------|-------------------------------------------------------------------------|
| `login_and_get_links.py`        | login to untapd website and retrive links for specified beer type photos.|
| `download.py`                   | uses the image linkes to go and download all the wanted photos.         |
| `sortToCupNoCup.py`             | using YOLO model to identify and crop out the beers for the photos      |
| `selfTest.py`                   | A simple GUI game to test human ability to solve our problem, used as a benchmark. |
| `fullModelTest.py`              | a script that pipelines the process from the uncropped untagged image input to a results with bbox and tags. |
| `BeerDL_50_augmented.ipynb`     | Jupyter notebook for barry's version of a beer classifier.                 |
| `BeerDL_lavie_model.ipynb`      | Jupyter notebook for barry's version of a beer classifier.                      |
| `EffnetAndDinoTraining.ipynb`   | Directory containing notebooks for training and evaluating models with motion blur noise. |
| `config!!!!11!!!!`                        | Directory containing environment configuration files.                   |
| `data`                          | Directory containing all the datasets and results                       |
| `HW`                            | Directory for our homework assigments.                                  |

## Installation Instructions

1. Get Anaconda with Python 3, follow the instructions according to your OS (Windows/Mac/Linux) at [link](https://www.anaconda.com/download)
2. Install the basic packages using the provided `environment.yml` file by running: `conda env create -f config/environment.yml` which will create a new conda environment named `deep_learn`. You can use `config/environment_no_cuda.yml` for an environment that uses pytorch cpu version.
3. Alternatively, you can create a new environment and install packages from scratch:
In Windows open `Anaconda Prompt` from the start menu, in Mac/Linux open the terminal and run `conda create --name deep_learn`. Full guide at [link](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)
4. To activate the environment, open the terminal (or `Anaconda Prompt` in Windows) and run `conda activate deep_learn`
5. Install the required libraries according to the table below (to search for a specific library and the corresponding command you can also look at [link](https://anaconda.org/))

### Libraries to Install

| Library | Installation Command |
|---------|----------------------|
| [numpy](https://numpy.org/) | `conda install -c conda-forge numpy` |
| [Pillow (PIL)](https://pillow.readthedocs.io/) | `conda install -c conda-forge pillow` |
| [tqdm](https://tqdm.github.io/) | `conda install -c conda-forge tqdm` |
| [scikit-learn](https://scikit-learn.org/stable/) | `conda install -c conda-forge scikit-learn` |
| [PyTorch](https://pytorch.org/) | `conda install pytorch -c pytorch` |
| [torchvision](https://pytorch.org/vision/stable/) | `conda install torchvision -c pytorch` |


## How to Use
After installing the required libraries, you can run the `main.ipynb` notebook to follow through with the project results and anlysis.

If you wish to train the models and evaluate them your self you can run the notebooks under `models_evaluation_with_noise` which contain the training and evaluation process for the all the models we analysed with the motion blur noise.

You can also download our weights from the following link:
- [Google Drive](https://drive.google.com/drive/folders/1Zj22MpCoxBWR9_azWvHRTWe_qfoa9Fsj?usp=drive_link)//barry link wieght 
  
  Just download the `models.zip` file and replace it with the existing `./data/models` folder in the repository.

##Gathering the Dataset
we could not fiind any tagged dataset that would help our project so we used avaliable resources and some deep learning models to create the needed dataset. 
we wrote a script(insert here) that downloads photos from the online website for beer lovers (untapped link), using the tags from the website itself as out correct labels( for example, we send the script to download all photos of a beer we know is a wheat beer, thus we created a "labeled" dataset.
next, since all the photos we user uploaded(social media like) we had a lot of irrelevant objects in the photos, so we used the pre-trained YOLO_v2 for its object detection and bbox features to get only the images of the beers themselves.
note that the YOLO model only had options for detecting cups or bottles, so we still had to manuelly delete a few dozen photos, but we used them for out control group as "not beer" label.
link to dataset:
<div align="center">
  <img src="./data/plots/random_images_from_train.png"/>
</div>

## Object Detection Models
In this project we trained the following SOTA object detection models:
//out_of_date
[torchvision models](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection):
- `ssd` [SSD: Single Shot MultiBox Detector](http://dx.doi.org/10.1007/978-3-319-46448-0_2)
- `faster rcnn` [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
- `retinanet` [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- `fcos` [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355)

[ultralytics yolo](https://docs.ultralytics.com/models/yolov8/):
- `yolov8m` [YOLOv8 models documentation](https://docs.ultralytics.com/models/yolov8/)

### Hyperparameter Tuning
A comprehensive search was performed across multiple hyperparameter spaces, including model preweight strategies, learning rates, momentum, and weight decay parameters.
The study involved carefully selecting combinations of hyperparameters to achieve the best validation score.
Below is a detailed table of the tuned hyperparameters.

| **Category**   | **Hyperparameter** | **Value / Range Used**               | **Description**                                   |
|----------------|--------------------|---------------------------------------|---------------------------------------------------|
| Model          | `preweight_mode`   | `random`                              | Model trained from scratch (no pretrained weights)|
| Training       | `batch_size`       | `64`                                  | Number of samples per training batch              |
| Training       | `epochs`           | `5` (search), `50` (final)            | Trials for tuning, then final full run            |
| Optimizer      | `optimizer`        | `AdamW`                               | Optimization algorithm                            |
| Adam/AdamW     | `lr`               | `[1e-4, 1e-3]` (log scale)            | Learning rate for AdamW (tuned)                   |
| Adam/AdamW     | `beta1`            | `0.9`                                 | Beta1 parameter (default)                         |
| Adam/AdamW     | `beta2`            | `0.999`                               | Beta2 parameter (default)                         |
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
graph TD
    A[Input 3×224×224] --> B[Conv 3→64<br/>BN + ReLU + MaxPool]
    B --> C[Conv 64→128<br/>BN + ReLU + MaxPool]
    C --> D[Conv 128→256<br/>BN + ReLU + MaxPool]
    D --> E[Conv 256→512<br/>BN + ReLU]
    E --> F[AdaptiveAvgPool 1×1]
    F --> G[Flatten]
    G --> H[Dropout(0.2)]
    H --> I[Linear 512→6<br/>Output logits]
```

<img src="./data/readme/barry_no_aug_results.png" width="400"/> 


*No Augmentations Test Accuracy: **72.30%** — Trained without data augmentations. Accuracy improves rapidly in the first 15 epochs and then plateaus around ~68–70%. The steady loss decrease indicates consistent learning, but the absence of augmentations likely limits the model’s ability to generalize to unseen variations.*

## Data Augmentations  
To improve robustness against variations in image orientation, perspective, and scale, we applied the following augmentations using [`torchvision.transforms`](https://pytorch.org/vision/stable/transforms.html):  

| **Augmentation**          | **Parameters**                                      | **Description**                                                                 |
|---------------------------|------------------------------------------------------|---------------------------------------------------------------------------------|
| `Resize`                  | `(256, 256)`                                        | Resize the input image to a fixed size.                                         |
| `RandomCrop`              | `224`                                               | Randomly crop the image to 224×224 pixels.                                      |
| `RandomHorizontalFlip`    | *(default p=0.5)*                                   | Flip the image horizontally with a probability of 0.5.                          |
| `RandomRotation`          | `20`                                                | Rotate the image randomly within ±20 degrees.                                   |
| `RandomAffine`            | `degrees=15`, `translate=(0.1, 0.1)`                | Apply random affine transformations with rotation and translation.              |
| `RandomPerspective`       | `distortion_scale=0.2`, *(default p=0.5)*           | Apply a random perspective transformation to simulate viewpoint changes.        |
| `ToTensor`                | —                                                   | Convert the image to a PyTorch tensor.                                          |
| `ToPILImage`              | —                                                   | Convert the tensor back to a PIL image (for further processing or visualization).|

These augmentations were chosen to improve the model ability to generalize to different viewing angles, making it more robust.  

## Results

### Barry's model results

<img src="./data/readme/barry_model_results.png" width="400"/> 

*Barry's Test Accuracy: **72.59%** — Accuracy improves steadily throughout training, reaching a plateau near the end. Loss decreases consistently, indicating stable learning without major overfitting. However, the final accuracy is lower compared to other approaches, suggesting that this configuration may lack sufficient capacity to capture all dataset-specific patterns.*


### Lavie's model results
<img src="./data/readme/lavie_model_results.png" width="400"/> 

*Lavie's Test Accuracy: **71.72%** — Accuracy improves quickly in the first 10 epochs, then gradually climbs toward ~72%. The steady decline in training loss suggests consistent learning without severe overfitting. However, the relatively early plateau in accuracy indicates that the model may be limited by its capacity or the features being used, preventing it from capturing all class distinctions effectively.*

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


## References

- [Untappd – Discover Beer](https://untappd.com) – Source for beer images and type labels.
- [YOLOv2 by Joseph Redmon](https://pjreddie.com/darknet/yolov2/) – Used for initial dataset object detection and cropping.
- [DINO (Facebook Research)](https://github.com/facebookresearch/dino) – Used as a benchmark model for comparison.
- [EfficientNet (TensorFlow Implementation)](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) – Used as a benchmark model for comparison.

