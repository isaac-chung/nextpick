from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from NextPick.image_search import *
from NextPick.plotly_map import create_plot, get_input_latlon, get_distances, get_top5_distance
import os
from config import DATA_FOLDER

# Create the application object
app = Flask(__name__)

# load index and model
input_dataset = ImageDataset('notebooks/data')
bs = 100
image_loader = torch.utils.data.DataLoader(input_dataset, batch_size=bs)
model, model_full = load_pretrained_model()

pd_files = input_dataset.get_file_df()
annoy_path = 'notebooks/annoy_idx.annoy'
if os.path.exists(annoy_path):
	annoy_idx_loaded = AnnoyIndex(512, metric='angular')
	annoy_idx_loaded.load(annoy_path)


@app.route('/')
def index():
	title_text = 'NextPick - by Isaac Chung'
	return render_template('index.html', title=title_text, username='chooch')

@app.route('/output')
def output():
	title_text = 'NextPick - by Isaac Chung'
	selection = request.args.get("selection")
	input_location = request.args.get("input_location")
	prox = request.args.get("prox")

	# Case if empty
	if selection != " ":
		print("..tag not empty")
		if selection == "ski":
			test_img = 'notebooks/ski-test-img.png'
			in_img = "assets/img/ski-test-img.png"
		elif selection == "war_mem":
			test_img = 'notebooks/test-img-war-mem.jpg'
			in_img = "assets/img/test-img-war-mem.jpg"
		elif selection == "banff":
			test_img = "static/assets/img/banff.jpg"
			in_img = "assets/img/banff.jpg"
		searches = eval_test_image(test_img, model, annoy_idx_loaded, top_n=60) # returns more than top 5 for processing
		df = create_df_for_map_plot(searches, pd_files)
		input_latlon = get_input_latlon(input_location)
		df = get_distances(input_latlon, df)

		df = get_top5_distance(df, prox)
		map_plot = create_plot(df, input_latlon)

		return render_template("results.html", title=title_text, flag="1", sel_input=selection,
							   df=df, plot=map_plot, input_location=input_location,
							   input_latlon=input_latlon, input_pic=in_img
							   )
	else:
		print("..tag empty")
	return render_template("index.html",
						   title=title_text, flag="0", sel_input=selection,
						   sel_form_result="Empty"
						   )


@app.route('/<path:filename>')
def download_file(filename):
	return send_from_directory(DATA_FOLDER, filename, as_attachment=True)


# start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=True) #will run locally http://127.0.0.1:5000/
