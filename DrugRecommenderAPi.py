from flask import Flask, request, jsonify
import pandas as pd
import pickle
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


model = pickle.load(open('finalized_model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            logging.error("No data provided")
            return jsonify({'error': 'No data provided'}), 400

        data_df = pd.DataFrame([data])
        data_preprocessed = preprocessor.transform(data_df)
        feature_names = strip_prefixes(preprocessor.get_feature_names_out())
        print(feature_names)
        data_preprocessed_df = pd.DataFrame(data_preprocessed.toarray(), columns=feature_names)
        predictions = model.predict(data_preprocessed_df)
        response = {'prediction': predictions.tolist()}
        return jsonify(response), 200
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500
def strip_prefixes(columns, prefix_list=['cat__', 'num__']):
    return [col.split('__')[-1] for col in columns if '__' in col for prefix in prefix_list if col.startswith(prefix)]


if __name__ == '__main__':
    app.run(debug=False)  # Set to False or manage via environment variable for production

