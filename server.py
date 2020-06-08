from flask import Flask, render_template, request, send_from_directory
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

	# Case if empty
	if selection == "ski":
		print("..ski tag")
		test_img = 'notebooks/ski-test-img.png'
		searches = eval_test_image(test_img, model, annoy_idx_loaded, top_n=30) # returns more than top 5 for processing
		df = create_df_for_map_plot(searches, pd_files)
		input_latlon = get_input_latlon(input_location)
		df = get_distances(input_latlon, df)
		df = get_top5_distance(df)
		bar = create_plot(df)

		return render_template("index.html", title=title_text,flag="1", sel_input=selection,
							   df=df, plot=bar, input_location=input_location,
							   input_latlon=input_latlon,
							   my_path='/ski resort/49788543373.jpg'
							   )
	elif selection == "war_mem":
		print("..war_mem tag")
		return render_template("index.html",
							   title=title_text,
							   flag="1",
							   sel_input=selection,
							   sel_result="memorialzzzz")
	else:
		print("..whatever")
		some_output = "yeay!"
	return render_template("index.html",
						   title=title_text,
						   flag="0",
						   some_output=some_output,
						   sel_input=selection,
						   sel_form_result="Empty")


@app.route('/<path:filename>')
def download_file(filename):
	return send_from_directory(DATA_FOLDER, filename, as_attachment=True)


# start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=True) #will run locally http://127.0.0.1:5000/
