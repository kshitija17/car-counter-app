from flask import Flask, request,app,jsonify,url_for,render_template,Response
import numpy as np
import run


app = Flask(__name__,static_url_path='/static')
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/car_counter', methods=['POST'])
def car_counter():
    video_url = request.form['videoUrl']
    # Create the response object with the appropriate MIME type
    return Response(run.run(video_url), mimetype='multipart/x-mixed-replace; boundary=frame')
    






if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=3000)