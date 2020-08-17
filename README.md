# fashion-image-inpainting
A Benchmark for Inpainting of Clothing Images with Irregular Holes

*Accepted to the European Conference on Computer Vision, ECCV 2020, Advances in Image Manipulation workshop and challenges on image and video manipulation [AIM2020](https://data.vision.ee.ethz.ch/cvl/aim20/)*

![Masked & Generated Images][fig1]

[fig1]: ./assets/fig1.png

## Project Details

Framework: PyTorch

Datasets: [FashionGen](https://fashion-gen.com), [FashionAI](https://tianchi.aliyun.com/markets/tianchi/FashionAI), [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html), [DeepFashion2](https://github.com/switchablenorms/DeepFashion2)

Masks: [qd-imd](https://github.com/karfly/qd-imd)

Base code: [birdortyedi](https://github.com/birdortyedi/description-aware-fashion-inpainting)

Hardware: 2x GTX 1080Ti or 2x Tesla V100

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

![Architecture][arch]

[arch]: ./assets/arch.png

#### Hyper-parameters

| Hyper-parameter        | Value         |
| -------------          |:-------------:|
| Optimizer              | Adam          |
| Betas                  | [0.5, 0.9]    |
| Generator LR           | 0.0002        |
| Discriminator LR       | 0.0001        |
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

#### Quantitative Results

![Quantitative][quantitative]

[quantitative]: assets/quantitative.png

#### Qualitative Results

![Qualitative][qualitative]

[qualitative]: assets/qualitative.png

#### Inpainting Clothings

![Inpainting][inpainting]

[inpainting]: assets/inpainting.png

#### Effect of dilation on Partial Convolutions

![Dilation][dilation]

[dilation]: assets/dilation.png

#### More Qualitative Results

![More Results][more]

[more]: assets/more.png

## Contacts

Please feel free to open an issue or to send an e-mail to `furkan.kinli@ozyegin.edu.tr`
