from flask import Flask, render_template, request
from chatbot import chatbot_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def get_response():
    user_input = request.args.get("msg")
    return chatbot_response(user_input)

if __name__ == "__main__":
    app.run(debug=True)
