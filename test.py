# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

cap = cv2.VideoCapture(0)

hands = mphands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Finger landmark IDs
TIP_IDS = [4, 8, 12, 16, 20]


def count_fingers(hand_landmarks, hand_label):
    fingers = []

    landmarks = hand_landmarks.landmark

    # Thumb
    # If thumb counting is reversed, swap < and > here
    if hand_label == "Right":
        if landmarks[4].x < landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if landmarks[4].x > landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Other 4 fingers
    # Tip higher than PIP joint means finger is up
    for tip_id in [8, 12, 16, 20]:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)


while True:
    data, image = cap.read()

    if not data:
        print("Camera not working")
        break

    image = cv2.flip(image, 1)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    total_fingers = 0
    hands_count = 0

    if results.multi_hand_landmarks:
        hands_count = len(results.multi_hand_landmarks)

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            hand_label = handedness.classification[0].label

            fingers_count = count_fingers(hand_landmarks, hand_label)
            total_fingers += fingers_count

            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mphands.HAND_CONNECTIONS
            )

            # Get hand position to write text near it
            h, w, c = image.shape
            wrist = hand_landmarks.landmark[0]
            x = int(wrist.x * w)
            y = int(wrist.y * h)

            cv2.putText(
                image,
                f"{hand_label}: {fingers_count}",
                (x - 40, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.putText(
        image,
        f"Hands: {hands_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.putText(
        image,
        f"Total Fingers: {total_fingers}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow('Handtracer', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()