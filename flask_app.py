from flask import Flask, request, jsonify
import mediapipe as mp
import cv2
import numpy as np

app = Flask(__name__)

pose_detection = mp.solutions.pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)


@app.route("/", methods=["POST"])
def get_belly_coordinates():
    # Retrieve the image from the POST request
    file = request.files["image"]
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Convert the image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image with the Pose Detection pipeline
    results = pose_detection.process(img)

    if results.pose_landmarks is not None:
        # Extract the relevant landmarks for the belly
        landmark_11 = results.pose_landmarks.landmark[11]
        landmark_12 = results.pose_landmarks.landmark[12]
        landmark_23 = results.pose_landmarks.landmark[23]
        landmark_24 = results.pose_landmarks.landmark[24]

        # Calculate the coordinates of the center of the belly
        x = int(
            (landmark_11.x + landmark_12.x + landmark_23.x + landmark_24.x)
            / 4
            * img.shape[1]
        )
        y = int(
            (landmark_11.y + landmark_12.y + landmark_23.y + landmark_24.y)
            / 4
            * img.shape[0]
        )

        # Return the coordinates as a JSON object
        return jsonify({"x": x, "y": y})

    # If no belly landmarks were detected, return an error message
    return jsonify({"error": "No belly detected."})


if __name__ == "__main__":
    app.run()
