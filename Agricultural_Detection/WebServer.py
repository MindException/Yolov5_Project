from flask import Flask, jsonify, request, render_template, make_response, send_file
from flask_cors import CORS

app = Flask(__name__, static_url_path="/static")
CORS(app)

@app.route("/")
def main():                           
    return render_template("Agricultural_products.html")

if __name__ == "__main__":
    # app.register_blueprint(Product_Detect.bp)              
    app.run(host="127.0.0.1", port="8123")