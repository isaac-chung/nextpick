# App path
APP_PATH = '/home/ubuntu/application'

# image preprocessing
RESIZE = (256, 256)
CROP = 224

# database
DATA_FOLDER = "%s/NextPick/data" %APP_PATH
BATCH = 100

# annoy index
ANNOY_PATH = '%s/NextPick/NextPick/annoy_idx.annoy' % APP_PATH
ANNOY_METRIC = 'angular'
ANNOY_TREE = 132

# ResNet18 Feature Embeddings
RESNET18_FEAT = 512

# places365
NUMCLASS = 365

# upload image
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# top_n images
TOP_N = 20

