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
python test_image.py

optional arguments:
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
<img src="https://github.com/Tagirov0/super-resolution/blob/main/test/result_srgan.jpg" width=100% height=100%>

* Srresnet: 300 epochs
* SRGAN 250 epochs with pretrained srresnet

Model            | SSIM      | PSNR 
---              |   ---     | ---   
SRGAN            |  0.86     | 29.75
SRRESNET         |  0.90     | 32.65 

### Hyperparameters
* batch_size = 64
* lr = 1e-4
* betas = (0.9, 0.999)
* image size = 96x96

## Perfomance
#### GPU: Tesla T4
It takes 6 seconds to process one image 1080x720

