# NextPick
Explore your next picks of travel locations by simply uploading
 a scenery image.
 
<p float="left">
    <img src="/static_img/49770197542.jpg" width="400"/>
    <img src="/static_img/49826303651.jpg" width="400"/>
</p>
     
NextPick uses pre-trained deep learning models on the 
[places365 dataset](https://github.com/CSAILVision/places365)
to generate image vector embeddings. These embeddings are
compared to geo-tagged images from [Flickr](https://www.flickr.com/).
Based on the image similarities and user-defined preferences,
the top 5 locations are provided. 
 
 ## Dependencies
Make sure all dependencies are installed from 'requirements.txt'. 
```
pip install -r requirements.txt
```

## Usage
![app](/static_img/app_screenshot_wk2.PNG)