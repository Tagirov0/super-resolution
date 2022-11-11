# Image Super-Resolution

## Usage
```
# clone repository
git clone https://github.com/Tagirov0/super-resolution.git

cd super-resolution

# install dependency
pip install -r requirements.txt
```

### Test image
```
python test_image.py --image_name IMAGE_NAME

arguments:
--image_name                  low resolution image name
--test_mode                   using GPU or CPU
```

### Train
* Install celeba [dataset](https://drive.google.com/file/d/1ScUq_VyL49cfYV8JMB2F7Ys14UQgWj5N/view?usp=share_link)
* Modify the [srgan_config.py](https://github.com/Tagirov0/super-resolution/blob/main/srgan_config.py) file
```
python train.py
```

## Result
<img src="https://github.com/Tagirov0/super-resolution/blob/main/test/result.jpg?raw=true" width=80% height=80%>

* Srresnet: 300 epochs
* SRGAN 250 epochs with pretrained srresnet

Model            | SSIM      | PSNR 
---              |   ---     | ---   
SRGAN            |  0.86     | 29.75
SRRESNET         |  0.90     | 32.65 

* The [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#:~:text=Peak%20signal%2Dto%2Dnoise%20ratio%20(PSNR)%20is%20an,the%20fidelity%20of%20its%20representation.) block computes the peak signal-to-noise ratio, in decibels, between two images. This ratio is used as a quality measurement between the original and a compressed image. The higher the PSNR, the better the quality of the compressed, or reconstructed image.

* The structural similarity index measure [SSIM](https://en.wikipedia.org/wiki/Structural_similarity#:~:text=The%20structural%20similarity%20index%20measure,the%20similarity%20between%20two%20images.) is a method for predicting the perceived quality of digital television and cinematic pictures, as well as other kinds of digital images and videos. SSIM is used for measuring the similarity between two images.

### Hyperparameters
* batch_size = 64
* lr = 1e-4
* betas = (0.9, 0.999)
* image size = 96x96

## Perfomance
#### GPU: Tesla T4
It takes 6 seconds to process one image 1080x720

