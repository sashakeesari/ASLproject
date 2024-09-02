import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Check the expected number of features for the model
expected_features = model.n_features_in_  # Get the expected number of features directly from the model

# Initialize the video capture (change index if necessary)
cap = cv2.VideoCapture(0)  # Change index if the wrong camera is selected

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: '0', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
               19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    # Capture a frame from the video
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret or frame is None or frame.size == 0:
        print("Failed to grab a frame from the camera.")
        continue

    # Get the frame's dimensions
    H, W, _ = frame.shape

    # Convert frame to RGB for Mediapipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,  # image to draw on
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract landmarks
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Normalize landmarks
            min_x = min(x_)
            min_y = min(y_)
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)

        # Ensure data_aux has the correct number of features
        if len(data_aux) < expected_features:
            # Pad with zeros if the number of features is less than expected
            data_aux.extend([0] * (expected_features - len(data_aux)))
        elif len(data_aux) > expected_features:
            # Truncate if there are too many features
            data_aux = data_aux[:expected_features]

        # Calculate bounding box
        x1 = max(int(min(x_) * W) - 10, 0)
        y1 = max(int(min(y_) * H) - 10, 0)
        x2 = min(int(max(x_) * W) + 10, W)
        y2 = min(int(max(y_) * H) + 10, H)

        # Predict the gesture using the model
        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
        except Exception as e:
            print(f"Prediction error: {e}")

    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
