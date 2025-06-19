import cv2
import dlib
import numpy as np
import pandas as pd
import os

# 얼굴 감지기와 랜드마크 예측기
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 입과 턱 관련 포인트 인덱스
MOUTH_POINTS = list(range(48, 68))
JAW_POINTS = list(range(6, 11))  # 턱 중앙 부위

# 특징 추출 함수
def extract_features(landmarks):
    # 입 좌우(x1)
    x1 = landmarks[54][0] - landmarks[48][0]
    # 입 상하(x2)
    x2 = landmarks[66][1] - landmarks[62][1]
    # 윗입술 두께(x3)
    x3 = landmarks[62][1] - landmarks[51][1]
    # 아랫입술 두께(x4)
    x4 = landmarks[57][1] - landmarks[66][1]
    # 턱 좌우(x5)
    x5 = landmarks[10][0] - landmarks[6][0]
    # 턱 높이(x6)
    x6 = landmarks[8][1] - landmarks[57][1]
    return [x1, x2, x3, x4, x5, x6]

# 저장 파일 초기화
csv_file = "features.csv"
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'label'])
    df.to_csv(csv_file, index=False)

# 웹캠 열기
cap = cv2.VideoCapture(1)

print("[S] 키: 촬영 | [Q]: 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # 랜드마크 그리기
        # 랜드마크 인덱스에 해당하는 점들을 그리고 싶다면:
        for idx in MOUTH_POINTS + JAW_POINTS:
            x, y = landmarks[idx]
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # 네모박스
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Capture Mouth Features", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # 촬영
        if faces:
            feats = extract_features(landmarks)
            label = input("입력한 모음(예: ㅏ, ㅓ, ㅗ...): ").strip()
            feats.append(label)
            df = pd.read_csv(csv_file)
            df.loc[len(df)] = feats
            df.to_csv(csv_file, index=False)
            print("저장 완료: ", feats)
        else:
            print("얼굴이 감지되지 않았습니다.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
