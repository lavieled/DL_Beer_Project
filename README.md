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

<div align="center">
  <img src="./data/plots/training_loss_val_map/training_loss_val_map.png"/>
</div>

## Data Augmentations
To cope with the motion blur noise, we applied the following data augmentations using [kornia](https://kornia.readthedocs.io/en/latest/):

| **Augmentation**       | **Parameters**                                                                 | **Description**                                                                 |
|------------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `RandomMotionBlur`     | `kernel_size=(3, 51)`, `angle=(-180.0, 180.0)`, `direction=(-1.0, 1.0)`, `p=0.4` | Apply random motion blur.                                                      |
| `RandomGaussianBlur`   | `kernel_size=(3, 3)`, `sigma=(0.1, 2.0)`, `p=0.3`                               | Apply random Gaussian blur to simulate out-of-focus blur.                      |
| `RandomSharpness`      | `sharpness=(0.5, 2.0)`, `p=0.3`                                                 | Adjust the sharpness of the image to simulate varying levels of focus.         |
| `ColorJiggle`          | `brightness=0.2`, `contrast=0.2`, `saturation=0.2`, `p=0.2`                      | Apply a random transformation to the brightness, contrast, saturation, and hue.|

These augmentations were chosen to help the model generalize better to different types of blur that might be encountered in real-world scenarios.

## Post Augmentations Results

### Results on the Clean Test Set

<div align="center">
  <img src="./data/plots/test_map_fps/clean_test_map_fps_with_aug.png" style="height: 500px;"/>
</div>

### Results on the Test Set Test Set with Uniform Motion Blur

<div align="center">
  <img src="./data/plots/test_map_fps/uniform_test_map_fps.png" style="height: 250px;"/>
</div>

### Results on the Test Set with Ellipse Motion Blur

<div align="center">
  <img src="./data/plots/test_map_fps/ellipse_test_map_fps.png" style="height: 250px;"/>
</div>

### Results on the Test Set with Natural Motion Blur

<div align="center">
  <img src="./data/plots/test_map_fps/natural_test_map_fps.png" style="height: 500px;"/>
</div>

## Potholes Severity

<div align="center">
  <img src="./data/plots/severity_checks/potholes_with_severity_images.png"/>
</div>

<div align="center">
  <img src="./data/plots/severity_checks/pothole_class_distribution.png"/>
</div>


## References

- [Untappd – Discover Beer](https://untappd.com) – Source for beer images and type labels.
- [YOLOv2 by Joseph Redmon](https://pjreddie.com/darknet/yolov2/) – Used for initial dataset object detection and cropping.
- [DINO (Facebook Research)](https://github.com/facebookresearch/dino) – Used as a benchmark model for comparison.
- [EfficientNet (TensorFlow Implementation)](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) – Used as a benchmark model for comparison.

