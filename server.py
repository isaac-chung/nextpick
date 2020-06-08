from flask import Flask, render_template, request, session, make_response
from NextPick.image_search import *
import os

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

	# Case if empty
	if selection == "ski":
		print("..ski tag")
		test_img = 'notebooks/ski-test-img.png'
		searches = eval_test_image(test_img, model, annoy_idx_loaded)
		df = create_df_for_map_plot(searches, pd_files)

		return render_template("index.html", title=title_text,flag="1",sel_input=selection,
							   cos=df['cos_diff'], address=df['address']
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


# start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=True) #will run locally http://127.0.0.1:5000/
