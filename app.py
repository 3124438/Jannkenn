import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import sys
import traceback

# --- 🧠 強制リセット機構 ---
# Streamlitが過去の壊れたMediaPipeを記憶している場合、ここで強制的に記憶を消去します
for key in list(sys.modules.keys()):
    if key.startswith('mediapipe'):
        del sys.modules[key]

# --- MediaPipeの安全な読み込み ---
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
except Exception as e:
    # もしエラーが出る場合は、表面的なAttributeErrorではなく「本当の原因」を画面に出力します
    st.error("⚠️ MediaPipeの読み込みでエラーが発生しました。以下のログをコピーして教えてください。")
    st.code(traceback.format_exc())
    st.stop()

# --- 設定 ---
CLASS_NAMES_JP = ['グー', 'チョキ', 'パー']
MODEL_FILENAME = 'my_janken_model.keras'

st.title("骨格推定じゃんけんAI ✊✌️✋")
st.write("Webカメラで手を撮影してください！")

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_FILENAME)
    except Exception:
        return None

model = load_model()

if model is None:
    st.error(f"⚠️ エラー: {MODEL_FILENAME} が見つかりません。")
else:
    # カメラウィジェット（写真撮影モード）
    camera_image = st.camera_input("ここをクリックして撮影")

    if camera_image is not None:
        # 画像データをOpenCV形式に変換
        bytes_data = camera_image.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        # MediaPipeで骨格推定
        with mp_hands.Hands(
            static_image_mode=True, # Streamlitは静止画なのでTrue
            max_num_hands=1,
            min_detection_confidence=0.5
        ) as hands:
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                data = []
                for landmark in hand_landmarks.landmark:
                    data.append(landmark.x)
                    data.append(landmark.y)

                input_data = np.array([data])
                prediction = model.predict(input_data, verbose=0)
                scores = prediction[0]
                
                # 結果の表示
                max_index = np.argmax(scores)
                st.success(f"判定結果: **{CLASS_NAMES_JP[max_index]}** ({scores[max_index]*100:.1f}%)")
                
                # 詳細な確率のバー表示
                st.write("--- 詳細 ---")
                for i, name in enumerate(CLASS_NAMES_JP):
                    st.progress(float(scores[i]), text=f"{name}: {scores[i]*100:.1f}%")
            else:
                st.warning("手が見つかりませんでした。もう一度、手をカメラの中央に映して撮影してください。")
