# fashion-image-inpainting
A Benchmark for Inpainting of Clothing Images with Irregular Holes

*Submitted to the European Conference on Computer Vision, ECCV 2020, Advances in Image Manipulation workshop and challenges on image and video manipulation [AIM2020](https://data.vision.ee.ethz.ch/cvl/aim20/)*

## Project Details

Framework: PyTorch

Datasets: [FashionGen](https://fashion-gen.com), [FashionAI](https://tianchi.aliyun.com/markets/tianchi/FashionAI), [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html), [DeepFashion2](https://github.com/switchablenorms/DeepFashion2)

Base code: [birdortyedi](https://github.com/birdortyedi/description-aware-fashion-inpainting)

Hardware: 2x GTX 1080Ti / 2x TITAN V100

``` git clone https://github.com/birdortyedi/fashion-image-inpainting.git```

## Requirements

* Python >= 3.6
* torch >= 1.2.0
* torchvision >= 0.4.2
* tensorboardX >= 1.9
* numpy >= 1.16
* pillow >= 6.2.0
* h5py >= 2.10.0
* tqdm >= 4.38.0
* colorama >= 0.4.1

## Run

``` python main.py ```

**TODO:** use argparser

## Implementation Details
#### Hyper-parameters

| Hyper-parameter        | Value         |
| -------------          |:-------------:|
| Optimizer              | Adam          |
| Generator LR           | 0.0004        |
| Discriminator LR       | 0.0002        |
| Scheduler              | Exponential   |
| Schedule Rate          | 0.9           |
| Batch Size             | 32            |
| Mask Form              | Free-form     |
| Multi-GPU              | True          |

#### Pre-processing

| Methods                | Range         |
| -------------          |:-------------:|
| Flipping               | Horizontal    |
| Resizing               | 256           |

## Results

## Contacts

Please feel free to open an issue or to send an e-mail to `furkan.kinli@ozyegin.edu.tr`
