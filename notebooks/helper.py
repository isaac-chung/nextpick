import config
import flickrapi
import requests
import os
import logging


def get_photos(image_tag):
    # extras = ','.join(SIZES)
    flickr = flickrapi.FlickrAPI(config.api_key, config.api_secret)
    photos = flickr.walk(text=image_tag,        # it will search by image title and image tags
                         extras='url_c',     # get the urls for each size we want
                         privacy_filter=1,   # search only for public photos
                         per_page=50,
                         sort='relevance')   # we want what we are looking for to appear first

    return photos


def get_urls(image_tag, max):
    photos = get_photos(image_tag)
    counter = 0
    urls = []

    for photo in photos:
        if counter < max:
            url = photo.get('url_c')
            if url:
                urls.append(url)
                counter += 1
            # if no url for the desired sizes then try with the next photo
        else:
            break
    return urls


def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def download_images(urls, path):
    create_folder(path)
    total = 0

    for url in urls:
        image_name = str(total).zfill(6) + '.jpg'
        image_path = os.path.join(path, image_name)

        if not os.path.isfile(image_path):  # ignore if already downloaded
            response = requests.get(url, stream=True)

            with open(image_path, 'wb') as outfile:
                outfile.write(response.content)

        total += 1

    logging.info('download completed.')