# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import signal
import sys
from types import FrameType
from flask import Flask, render_template, request, jsonify
from utils.logging import logger
from flask_cors import CORS
from document_retriever import retrieve_answers

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    # Use basic logging with custom fields
    logger.info(logField="custom-entry", arbitraryField="custom-entry")

    # https://cloud.google.com/run/docs/logging#correlate-logs
    logger.info("Child logger with trace Id.")

    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def get_answer():
    # Log the received data
    logger.info("Received data: %s", request.get_json())

    # Get the question from the AJAX request
    question = request.get_json().get('question', '')

    # Hardcoded example answer
    #answer = "This is the answer to your question: {}".format(question)
    answers = retrieve_answers(question)

    # Return the most relevant answer (answers[0]) as JSON
    response = jsonify({'answer': answers[0]})
    logger.info("Sending response: %s", response.get_json())
    return response


def shutdown_handler(signal_int: int, frame: FrameType) -> None:
    logger.info(f"Caught Signal {signal.strsignal(signal_int)}")

    from utils.logging import flush

    flush()

    # Safely exit program
    sys.exit(0)


if __name__ == "__main__":
    # Running application locally, outside of a Google Cloud Environment

    # handles Ctrl-C termination
    signal.signal(signal.SIGINT, shutdown_handler)

    app.run(host="localhost", port=8080, debug=True)
else:
    # handles Cloud Run container termination
    signal.signal(signal.SIGTERM, shutdown_handler)
