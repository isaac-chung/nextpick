import argparse
from time import time
import secrets
import flickrapi
import requests
import os
import pandas as pd
import pickle
from PIL import Image


def get_photos(image_tag, path='data', verbose=1):
    # setup dataframe for data
    raw_photos = pd.DataFrame(columns=['latitude', 'longitude', 'farm', 'server', 'id', 'secret'])

    # initialize api
    flickr = flickrapi.FlickrAPI(secrets.api_key, secrets.api_secret, format='parsed-json')

    errors = ''
    try:
        # search photos based on settings
        photos = flickr.photos.search(tags=image_tag,
                                      sort='relevance',
                                      accuracy=11,  # city level
                                      content_type=1,  # photos only
                                      extras='description,geo,url_c',
                                      has_geo=1,
                                      geo_context=2,  # outdoors
                                      per_page=100,
                                      page=1
                                      )

        # append photo details: description and getags
        raw_photos = raw_photos.append(pd.DataFrame(photos['photos']['photo'])
                                       [['latitude', 'longitude', 'farm', 'server', 'id', 'secret']],
                                       ignore_index=True)

        # construct url from pieces
        raw_photos['url'] = 'https://farm' + raw_photos.farm.astype(
            str) + '.staticflickr.com/' + raw_photos.server.astype(str) + '/' + raw_photos.id.astype(
            str) + '_' + raw_photos.secret.astype(str) + '.jpg'

        # need a try/except here for images less than 'per page'
        if verbose:
            print('..downloading photos')
        download_images(raw_photos, image_tag, verbose)

        # save data
        if verbose:
            print('..saving metadata')
        with open('%s/%s/%s.pkl' % (path, image_tag, image_tag), 'wb') as f:
            pickle.dump(raw_photos, f)
            f.close()

        del raw_photos

    except:
        if verbose:
            print('Could not get info for: %s. ' % image_tag)
        errors = image_tag

    return errors


def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def download_images(df, keyword, verbose=1):
    path = ''.join(['data/', keyword])
    create_folder(path)

    if verbose:
        print('...df length: %d' % len(df.index))
        print('...going through each row of dataframe')
    for idx, row in df.iterrows():
        try:
            image_path = ''.join([path, '/', row.id, '.jpg'])
            response = requests.get(row.url)  # , stream=True)

            with open(image_path, 'wb') as outfile:
                outfile.write(response.content)
                outfile.close()

        except:
            if verbose:
                print('...Error occured at idx: %d' % idx)

    if verbose:
        print('...download completed.')


def get_places(path='../places365/IO_places365.txt', verbose=1):

    places = pd.read_csv(path, sep=' ', names=['key', 'label'])
    # retrieve all outdoor scene categories. We clean up the 'key' column, remove duplicates,
    # and re-index the dataframe.
    places['key'] = places['key'].str[3:].str.split('/', 1, expand=True)
    places = places[places.label == 2]
    places = places.drop_duplicates(ignore_index=True)
    places['key'] = places['key'].str.strip('\'')
    places['key'] = places['key'].replace(to_replace='_', value=' ', regex=True)
    if verbose:
        print(places.count())  # should have 199

    return places


def run_flickr_download(places, data_folder_path='data', verbose=1):
    errors = []
    for idx, row in places.iterrows():

        # change this idx when it crashes. It will give an error for a few indices.
        # It probably means Flickr does not have geotagged images for these keywords.
        # We skip over those. Should have a total of 130 keywords at the end.
        if idx < 0:
            pass
        else:
            start = time()
            error = get_photos(row.key, data_folder_path, verbose)
            end = time()
            if verbose:
                print('%20s in %.0f seconds.' % (row.key, end - start))
                # should vary between 3-8 seconds depending on the keyword.

            if error != '':
                errors.append(error)


if __name__ == '__main__':

    """
    Run with the following command in shell:
    python flickr_images_download.py [--verbose] [--path]
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', type=int, choices=[0, 1], default=1,
                        help='output verbosity (0 or 1')
    parser.add_argument('-p', '--path', type=str, default='../NextPick-app/NextPick/data',
                        help='path of the data folder')
    args = parser.parse_args()

    places = get_places(verbose=args.verbose)
    run_flickr_download(places, data_folder_path=args.path, verbose=args.verbose)

    #  confirm that the folders work. showing the first image in basilica.
    test_keyword = 'basilica'
    default_path = '../NextPick-app/NextPick/data'
    with open('%s/%s/%s.pkl' % (default_path, test_keyword, test_keyword), 'rb') as f:
        test = pickle.load(f)
        f.close()
    image = Image.open('%s/%s/%s.jpg' % (default_path, test_keyword, test.id[0]))
    image.show()

