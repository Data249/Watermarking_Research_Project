# Watermarking_Research_Project

## How to run the program 


For Embedding :  python Embedding.py 

For Extraction:  python Extraction.py

## Install :

pip install -r requirements.txt or pip3 install -r requirements.txt

## Information :

 ```
Please read the documentation for a detailed understanding of implementations
```
  
  It is preferable to use virtualenv which allows you to avoid installing Python packages globally. (It is already available in this)
  

## Processing steps :

1. Place the original images in the source folder to resize the pixels.
2. Run the image.py file and the output will be saved at the destination folder in the format small_filename.
3.  Copy the images in the destination folder and paste them in coverimages folder for Embedding.
4. Run the Embedding.py file and choose the watermarking system you want to perform.
5. The Embedded watermarked image will be saved at the watermarked folder.
6. For Extraction, run the Extraction.py file and choose the Watermark system.
7. The Extracted watermark will be saved at the watermark_recovered folder.
8. The PSNR and MSE values are saved in the excels for both Embedding and Extraction.


## Folders Information :

1. coverimages: The images are used for the Embedding process.
2. destination: The resized images are stored.
3. key: The key used in the Embedding process is saved.
4. source: The original images are placed here for the resize process.
5. watermark_recovered: The extracted watermark images are saved.
6. watermarked: The embedded watermarked images are saved.  

  ![](readmegif.gif)
  
  
## Features

Six watermarking systems are implemented. They are:

1.DWT

2.DCT

3.DFT

4.SVD

5.DWT-SVD

6.SVD-DCT-DWT
 
Analyzing the watermarking systems using PSNR and MSE values.

  
If you have any doubts or facing issues, please contact us
  
  * Thrivikram Gujarathi (thrivikramlycan@gmail.com)

  * Siva Bathala (shivachowdary511@gmail.com)

  * Arup Mazumder (arupseu@gmail.com)
  
