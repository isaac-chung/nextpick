from flask import Flask, render_template, request, session, make_response
from processing import process_data

# Create the application object
app = Flask(__name__)

@app.route('/')
def index():
	title_text = 'NextPick - by Isaac Chung'
	return render_template('index.html', title=title_text, username='chooch')

# @app.route('/output', methods=['POST'])
# def output():
#    	# Pull input
# 	some_input = request.args.get('test')
#
# 	# Case if empty
# 	if some_input == "":
# 		return render_template("templates/index.html",
# 							   my_input=some_input,
# 							   my_form_result="Empty")
# 	else:
# 		some_output = "yeay!"
# 	some_number = 3
# 	some_image = "static/assets/img/ipad.jpg"
# 	return render_template("index.html",
# 						   my_input=some_input,
# 						   my_output=some_output,
# 						   my_number=some_number,
# 						   my_img_name=some_image,
# 						   my_form_result="NotEmpty")


# start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=True) #will run locally http://127.0.0.1:5000/
	# app.config["SECRET_KEY"] = "lkmaslkdsldsamdlsdmasldsmkdd"
