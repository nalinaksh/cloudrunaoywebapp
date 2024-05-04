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
from document_retriever import retrieve_answers, consult
from question_recommendation import recommend

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    # Use basic logging with custom fields
    logger.info(logField="custom-entry", arbitraryField="custom-entry")

    # https://cloud.google.com/run/docs/logging#correlate-logs
    logger.info("Child logger with trace Id.")

    return render_template('index.html')

@app.route('/consult')
def consult():
    # Use basic logging with custom fields
    logger.info(logField="custom-entry", arbitraryField="custom-entry")

    # https://cloud.google.com/run/docs/logging#correlate-logs
    logger.info("Child logger with trace Id.")

    return render_template('consult.html')


@app.route('/answer', methods=['POST'])
def get_answer():
    # Log the received data
    logger.info("Received data: %s", request.get_json())

    # Get the question from the AJAX request
    question = request.get_json().get('question', '')

    # Hardcoded example answer
    #answer = "This is the answer to your question: {}".format(question)
    gita_res, aoy_res = retrieve_answers(question)
    recs = recommend(question)
    str1 = ""
    str2 = ""
    str3 = ""
    str4 = ""
    str5 = ""
    str6 = ""
    ans = ""
    
    if gita_res['Score'] >= 0.55:
        str1 = "<span style='font-size: 12px; color: maroon'>" + "Bhagvad Gita Chapter " + str(gita_res['Chapter']) + " " + "Verse " + str(gita_res['Verse']) + "</span>" + "<br><br>"
        str2 = "<span style='font-weight: bold'>" + gita_res['Sanskrit'] + "</span>" + "<br><br>"
        str3 = "<span>" + gita_res['English'] + "</span>" + "<br><br>"
        
    if aoy_res['Score'] >= 0.55:
        str4 = "<span style='font-size: 12px; color: maroon'>" + "Autobiography " + aoy_res['Chapter'] + "</span>" + "<br><br>"
        str5 = "<span>" + aoy_res['Chunk Content'] + "</span>" + "<br><br>"

    if gita_res['Score'] >= 0.55 or aoy_res['Score'] >= 0.55:
        str6 = "<span style='font-size: 16px; color: grey'>" + "You may also like to ask:" + "<br>" \
        + "<i>" + recs[0] + "</i>" + "<br>" \
        + "<i>" + recs[1] + "</i>" + "<br>" \
        + "<i>" + recs[2] + "</i>" + "<br>" \
        + "<i>" + recs[3] + "</i>" \
        + "</span>"

    if gita_res['Score'] < 0.55 and aoy_res['Score'] < 0.55:
        ans = "<span>" + "Your query did not fetch any relevant results" + "</span>" + "<br><br>"
    else:
        ans =  str1 + str2 + str3 + str4 + str5 + str6
    # Return the most relevant answer (answers[0]) as JSON
    # response = jsonify({'answer': answers[0], 'chapter': chapters[0]})
    response = jsonify({'answer': ans})
    logger.info("Sending response: %s", response.get_json())
    return response

@app.route('/get_consultation', methods=['GET','POST'])
def consult():
    # Log the received data
    logger.info("Received data: %s", request.get_json())

    # Get the question from the AJAX request
    question = request.get_json().get('question', '')

    gita_counsel = consult(question)
    ans = "<span>" + gita_counsel + "</span>"
    
    response = jsonify({'answer': ans})
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
