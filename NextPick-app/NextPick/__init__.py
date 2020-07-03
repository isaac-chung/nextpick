from flask import Flask
app = Flask(__name__)

from . import server
from NextPick.NextPick import image_search
from NextPick.NextPick import ImageDataset
from NextPick.NextPick import plotly_map