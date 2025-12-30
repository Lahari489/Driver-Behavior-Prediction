import serial
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np

# ---------- EAR (Eye Aspect Ratio) ----------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ---------- MAR (Mouth Aspect Ratio) ----------
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # 51, 59
    B = distance.euclidean(mouth[4], mouth[8])   # 53, 57
    C = distance.euclidean(mouth[0], mouth[6])   # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

# ---------- Serial Communication ----------
arduino = serial.Serial('COM3', 9600)  # Replace COM3 with your Arduino port

# ---------- Thresholds ----------
EYE_THRESH = 0.25
MOUTH_THRESH = 0.5
YAW_THRESH = 20   # degrees
ROLL_THRESH = 20  # degrees
FRAME_LIMIT = 30  # number of continuous frames before alert

# ---------- Face detection ----------
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

# ---------- Flags / Counters ----------
eye_flag = 0
mouth_flag = 0
head_flag = 0

# ---------- 3D Model points for Head Pose ----------
model_points = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye left corner
    (225.0, 170.0, -135.0),    # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)    # Right mouth corner
], dtype="double")

# ---------- Main Loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # EAR (Eye)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # MAR (Mouth)
        mouth = shape[48:68]
        mar = mouth_aspect_ratio(mouth)

        # Draw contours
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

        # ---------- EAR Check ----------
        if ear < EYE_THRESH:
            eye_flag += 1
            if eye_flag >= FRAME_LIMIT:
                cv2.putText(frame, "DROWSY ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                arduino.write(b'1')  # Drowsy alert
        else:
            eye_flag = 0

        # ---------- MAR Check ----------
        if mar > MOUTH_THRESH:
            mouth_flag += 1
            if mouth_flag >= FRAME_LIMIT:
                cv2.putText(frame, "YAWNING ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                arduino.write(b'2')  # Yawning alert
        else:
            mouth_flag = 0

        # ---------- Head Pose Estimation ----------
        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        image_points = np.array([
            shape[30],  # Nose tip
            shape[8],   # Chin
            shape[36],  # Left eye left corner
            shape[45],  # Right eye right corner
            shape[48],  # Left mouth corner
            shape[54]   # Right mouth corner
        ], dtype="double")

        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        rmat, _ = cv2.Rodrigues(rotation_vector)
        proj_mat = np.hstack((rmat, translation_vector))
        _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_mat)
        pitch, yaw, roll = [angle[0] for angle in eulerAngles]

        # ---------- Head Pose Check ----------
        if abs(yaw) > YAW_THRESH or abs(roll) > ROLL_THRESH:
            head_flag += 1
            if head_flag >= FRAME_LIMIT:
                cv2.putText(frame, "HEAD TURN ALERT!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                arduino.write(b'3')  # Head turn alert
        else:
            head_flag = 0

        # ---------- Display Ratios Continuously ----------
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"YAW: {yaw:.2f}", (300, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"ROLL: {roll:.2f}", (300, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show Frame
    cv2.imshow("Driver Monitor", frame)

    # Quit on "q"
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# ---------- Cleanup ----------
cap.release()
cv2.destroyAllWindows()
arduino.close()
