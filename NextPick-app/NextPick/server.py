from flask import Flask, render_template, request, send_from_directory
from annoy import AnnoyIndex
from NextPick.NextPick.image_search import load_pretrained_model, eval_test_image, create_df_for_map_plot
from NextPick.NextPick.ImageDataset import ImageDataset
from NextPick.NextPick.plotly_map import create_plot, get_input_latlon, get_distances, get_top5_distance
import os
import torch
from base64 import b64encode
import pickle
from NextPick import app
import NextPick.config as cfg


# Create the application object
# app = Flask(__name__)
app.secret_key = 'random'

# load index and model
input_dataset = ImageDataset(cfg.DATA_FOLDER)
image_loader = torch.utils.data.DataLoader(input_dataset, batch_size=cfg.BATCH)
model, model_full = load_pretrained_model()
fname_df = '%s/NextPick/NextPick/pd_files.pkl' % cfg.APP_PATH
with open(fname_df, 'rb') as f:
	pd_files = pickle.load(f)
	f.close()
# pd_files = input_dataset.get_file_df()
annoy_path = cfg.ANNOY_PATH
if os.path.exists(annoy_path):
	annoy_idx_loaded = AnnoyIndex(cfg.RESNET18_FEAT, metric=cfg.ANNOY_METRIC)
	annoy_idx_loaded.load(annoy_path)
	print('Loaded annoy tree from memory.')


@app.route('/')
def index():
	title_text = 'NextPick - by Isaac Chung'
	return render_template('index.html', title=title_text, username='chooch')


@app.route("/image_upload", methods=['GET', 'POST'])
def upload_img():
	title_text = 'NextPick - by Isaac Chung'
	selection = request.form.get("selection")
	input_location = request.form.get("input_location")
	prox = request.form.get("prox")

	if request.method == 'POST':
		file = request.files['fileupload']
		if file.filename == '':
			print('..No selected file, but tag not empty')
			input_type = 'preselect'
			if selection == "ski":
				test_img = '%s/NextPick/static/assets/img/ski-test-img.png' % cfg.APP_PATH
				in_img = 'assets/img/ski-test-img.png'
			elif selection == "venice":
				test_img = '%s/NextPick/static/assets/img/venice.jpg' % cfg.APP_PATH
				in_img = "assets/img/venice.jpg"
			elif selection == "banff":
				test_img = "%s/NextPick/static/assets/img/banff.jpg" % cfg.APP_PATH
				in_img = "assets/img/banff.jpg"
		if file:
			print('.. using uploaded image')
			input_type = 'upload'
			test_img = file
			encoded = b64encode(file.read()) # encode the binary
			mime = "image/jpg"
			in_img = "data:%s;base64,%s" %(mime, encoded.decode()) # remember to decode the encoded data


		searches = eval_test_image(test_img, model, annoy_idx_loaded, top_n=cfg.TOP_N)
		df = create_df_for_map_plot(searches, pd_files) # 0.5s per 1 top_n
		input_latlon = get_input_latlon(input_location)
		df = get_distances(input_latlon, df)

		df = get_top5_distance(df, prox)
		map_plot = create_plot(df, input_latlon)

		return render_template("results.html", title=title_text, flag="1",
							   df=df, plot=map_plot, input_location=input_location,
							   input_latlon=input_latlon, input_pic=in_img, input_type=input_type
							   )
	else:
		render_template("index.html", title=title_text, flag="0",
						sel_input=selection, sel_form_result="Empty")


@app.route('/<path:filename>')
def download_file(filename):
	return send_from_directory(cfg.DATA_FOLDER, filename, as_attachment=True)
