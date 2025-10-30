from flask import Flask, render_template, Response, request, jsonify
import cv2
import pickle
import mediapipe as mp
import numpy as np
import base64
import gdown  # ← add this import
app = Flask(__name__)


# 🔽 ADD THESE LINES just before loading the model
url = "https://drive.google.com/uc?id=1yFVu1A-JlVpZBQtbQYmjop_PisZd6JJQ"  # replace with your real file ID
gdown.download(url, "model.h5", quiet=False)
# Load your model
model_dict = pickle.load(open('model.h5', 'rb'))
model = model_dict['model']

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: '', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G',
    8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N',
    15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U',
    22: 'V', 23: 'X', 24: 'Y', 25: 'Z'
}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        # Decode base64 image
        img_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        data_aux = []
        x_ = []
        y_ = []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        predicted_character = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            # Pad or trim to match model input
            if len(data_aux) < 84:
                data_aux += [0] * (84 - len(data_aux))
            elif len(data_aux) > 84:
                data_aux = data_aux[:84]

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = str(prediction[0])

        return jsonify({'prediction': predicted_character})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
