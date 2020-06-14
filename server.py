from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from NextPick.image_search import *
from NextPick.ImageDataset import ImageDataset
from NextPick.plotly_map import create_plot, get_input_latlon, get_distances, get_top5_distance
import os
import config as cfg
from base64 import b64encode


# Create the application object
app = Flask(__name__)

# load index and model
input_dataset = ImageDataset(cfg.DATA_FOLDER)
image_loader = torch.utils.data.DataLoader(input_dataset, batch_size=cfg.BATCH)
model, model_full = load_pretrained_model()

pd_files = input_dataset.get_file_df()
annoy_path = cfg.ANNOY_PATH
if os.path.exists(annoy_path):
	annoy_idx_loaded = AnnoyIndex(cfg.RESNET18_FEAT, metric=cfg.ANNOY_METRIC)
	annoy_idx_loaded.load(annoy_path)


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
				test_img = 'notebooks/ski-test-img.png'
				in_img = "assets/img/ski-test-img.png"
			elif selection == "venice":
				test_img = 'notebooks/venice.jpg'
				in_img = "assets/img/venice.jpg"
			elif selection == "banff":
				test_img = "static/assets/img/banff.jpg"
				in_img = "assets/img/banff.jpg"
		if file:
			print('.. using uploaded image')
			input_type = 'upload'
			test_img = file
			encoded = b64encode(file.read()) # encode the binary
			mime = "image/jpg"
			in_img = "data:%s;base64,%s" %(mime, encoded.decode()) # remember to decode the encoded data


		searches = eval_test_image(test_img, model, annoy_idx_loaded, top_n=15)
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


# start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=True) #will run locally http://127.0.0.1:5000/

