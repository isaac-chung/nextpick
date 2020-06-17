# NextPick
*Pick your next insta-vacation*

Website: [nextpick.live](http://nextpick.live)

<p float="left">
    <img src="/static_img/49770197542.jpg" height="250"/>
    <img src="/static_img/49826303651.jpg" height="250"/>
</p>

## What is NextPick?
There is an increasing portion of travellers saying that they use social media for planning their next trip, giving
a rise to the trend of social media-inspired tourism in recent years. These social media-inspired travellers tend
to go on "insta-holidays" (picturesque places), and "mini-vacations" (short duration). Closer-by locations are also 
favoured for travel time considerations. Combining the above creates the need for media and proximity driven searches 
for travel locations.

NextPick is my vision for addressing this need. It is a tool that recommends vacation spots based on an input image.
Similar images, along with their locations, are returned based on cosine similarity between the images and geo-proximity
to the user.

     
NextPick uses pre-trained deep learning models on the [places365 dataset](https://github.com/CSAILVision/places365)
to generate image feature embeddings. These embeddings are compared to geo-tagged images from 
[Flickr](https://www.flickr.com/). Based on the image similarities and user-defined preferences, the top 5 locations 
are provided. 
 

## How to use NextPick
The landing page allows the user to choose an image, enter their current 
location, and select a proximity preference.
![app](/static_img/wk3_screenshots/landingpage.jpg) 

After an image is selected and the 'Go' button is clicked, the selected image
and the entered location are shown.
![app](/static_img/wk3_screenshots/image1.jpg)

 This is followed by 5 images, along with 
their cosine difference to the input image and the distance to the current location.
![app](/static_img/wk3_screenshots/image2.jpg)
![app](/static_img/wk3_screenshots/image3.jpg)

And finally, the locations of those images are visualized on a map. Labels for each
data point is shown on hover. 
![app](/static_img/wk3_screenshots/map1.jpg)
![app](/static_img/wk3_screenshots/map3.jpg)

The user can zoom in and out of the Plotly map
![app](/static_img/wk3_screenshots/map2.jpg)


## Local setup
To run a local copy of NextPick, clone the master branch of the repository with
```
git clone https://github.com/isaac-chung/insight.git
```
Then, create a virtual environment and have all the dependencies in `requirements.txt` installed. With Anaconda3,
we can use the following to create one called `insight` with Python 3.7 and all packages installed.
```
conda create -n insight python=3.7
pip install r requirements.txt
```
I am currently keeping the image data on this [repository](https://github.com/isaac-chung/insight-image-data).
Once it's downloaded, place the folder at the root level of the insight project folder, and rename it as `data`.
If this link is broken, you can follow 
[this Jupyter notebook](https://github.com/isaac-chung/insight/blob/master/notebooks/1-flickr_api_images_geotag_download.ipynb) 
to retrieve your own geotagged imaged data using the Flickr API.
Also, in `config.py`, change the variable `DATA_FOLDER` to the path of the data folder you just created.

The app can be run locally using
```
python server.py
```
Navigate via your web browser to http://127.0.0.1:5000/. This version of the app is tested on Windows 10 and Ubuntu 
18.04 (WSL). 