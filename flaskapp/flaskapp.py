from flask import Flask, render_template
import namecnn
import tf_predict
import numpy as np
from tools import *

app = Flask(__name__)

@app.route('/name/<input_str>')
def predict(input_str):
    return namecnn.predict(input_str)


@app.route('/name_tf/<input_str>')
def predict2(input_str):
    return tf_predict.predict(input_str)


if __name__ == "__main__":
    app.run()
            
