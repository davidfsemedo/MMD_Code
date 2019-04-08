from run_predictions import get_prediction
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def parse_response(response):
    """
    Given a bot response (example: "</s>Example response</e></e></e></e></e></e>")
    parse it removing the start-token "</s>" and end-token "</e>"
    :param response: The bot response to be parsed
    :return: The parsed bot response
    """

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
    """
    Input API to start the process. Listens to endpoint "/" and expects to receive a user query under the "q" query parameter
    :return: The bot response for the given user query
    """
    user_query = request.args.get("q")

    # Retrieve the unprocessed bot response for the user query
    bot_response = get_prediction(user_query=user_query)

    # Parse the response and return it
    return parse_response(bot_response)


if __name__ == '__main__':
    app.run(port=9191, host="compute-0-1")

# Example Request:
# GET http://compute-0-1:9191/q?blue bag
