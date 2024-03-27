from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"


@app.route("/inside")
def inside():
    return "Hello World! from inside"

if __name__ == "__main__":
    app.run(debug=True)