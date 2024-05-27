from flask import Flask, request, jsonify
import pandas as pd
import json
import os
from event_detection_model import detect_events
import logging
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

def process_json_file(file):
    try:
        print("hereeeeb")
        json_data = file.readlines()
        print("hereeee34")
        json_objects = [json.loads(obj.strip()) for obj in json_data]
        print("hereeee")
        for obj in json_objects:
            obj['user'] = obj['user']['screen_name']
        df = pd.DataFrame(json_objects)
        print("hereeee2")
        df.drop(columns=['id_str'], inplace=True)
        print("hereeee3")
        df.rename(columns={'created_at': 'TweetDate', 'text': 'TweetText'}, inplace=True)

        return df
    except Exception as e:
        raise ValueError(f"Error processing JSON file: {str(e)}")

@app.route('/upload', methods=['POST'])
def upload_file():
    app.logger.info('Upload endpoint called')
    if 'file' not in request.files:
        app.logger.error('No file part in request')
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        app.logger.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.json'):
        try:
            app.logger.info('Processing JSON file')

            df = process_json_file(file)

            numEvents = int(request.form.get('numEvents', 5)) 

            events = detect_events(df, numEvents=numEvents)
            return jsonify({'events': events})
        except Exception as e:
            app.logger.exception('Error processing file')
            return jsonify({'error': str(e)}), 500

    app.logger.error('Invalid file format')
    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)
