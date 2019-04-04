from run_predictions import get_prediction
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def parse_response(response):
    # If the response is not empty, process it
    if response:
        response = response.split(" ")

        # Discard everything after the end token ("</e>")
        parsed_sentence = response[:response.index("</e>")]

        # Remove the start token
        parsed_sentence.remove("</s>")
        return " ".join(parsed_sentence)

    else:
        "I'm sorry, I didn't understand that."


@app.route("/")
def get_server_response():
    user_query = request.args.get("q")
    bot_response = get_prediction(user_query=user_query)
    return parse_response(bot_response)


if __name__ == '__main__':
    app.run(port=9191, host="compute-0-1")
