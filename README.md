# NextPick
Explore your next picks of travel locations by simply uploading
 a scenery image.
 
<p float="left">
    <img src="/static_img/49770197542.jpg" height="250"/>
    <img src="/static_img/49826303651.jpg" height="250"/>
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
![app](/static_img/wk2_screenshots/landingpage.PNG)
The landing page allows the user to choose an image and entered their current 
location. 
![app](/static_img/wk2_screenshots/image1.PNG)
After an image is selected and the 'Go' button is clicked, the selected image
and the entered location are shown. This is followed by 5 images, along with 
their cosine difference to the input image and the distance to the current location.
![app](/static_img/wk2_screenshots/image2.PNG)
![app](/static_img/wk2_screenshots/image3.PNG)
And finally, the locations of those images are visualized on a map.
![app](/static_img/wk2_screenshots/bottom.PNG)