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
--image_name                  test low resolution image name
--test_mode                   using GPU or CPU [default value is 'GPU'](choices:['GPU', 'CPU'])
```

### Train
* Install celeba [dataset](https://drive.google.com/file/d/1ScUq_VyL49cfYV8JMB2F7Ys14UQgWj5N/view?usp=share_link)
* Modify the [srgan_config.py](https://github.com/Tagirov0/super-resolution/blob/main/srgan_config.py) file
```
python train.py
```

## Result
<img src="https://github.com/Tagirov0/super-resolution/blob/main/test/result_srgan.jpg" width=100% height=100%>

## Perfomance
GPU: Tesla T4
Single image

