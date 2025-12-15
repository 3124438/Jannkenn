import gradio as gr
import tensorflow as tf
import numpy as np
import mediapipe as mp
import cv2
import os

# --- 設定 ---
CLASS_NAMES_JP = ['グー', 'チョキ', 'パー']
MODEL_FILENAME = 'my_janken_model.keras'
# ---

# MediaPipeの準備
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# モデルの読み込み
try:
    model = tf.keras.models.load_model(MODEL_FILENAME)
    print("✅ モデル読み込み完了！")
except Exception as e:
    print(f"⚠️ エラー: {MODEL_FILENAME} が見つかりません。")
    model = None 

# 予測関数
def classify_hand(image):
    if model is None:
        return {"モデル読込エラー": 0.0}
        
    if image is None:
        return None

    # 画像処理
    # Gradioから来る画像はすでにRGBになっていることが多いですが、
    # エラー防止のためそのまま処理に回します
    results = hands.process(image)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        data = []
        for landmark in hand_landmarks.landmark:
            data.append(landmark.x)
            data.append(landmark.y)

        input_data = np.array([data])

        prediction = model.predict(input_data, verbose=0)
        scores = prediction[0]

        result = {CLASS_NAMES_JP[i]: float(scores[i]) for i in range(len(CLASS_NAMES_JP))}
        return result

    else:
        return {"手が見つかりません": 0.0}

# Gradioインターフェース
iface = gr.Interface(
    fn=classify_hand,
    inputs=gr.Image(
        sources=["webcam"],
        label="Webカメラ",
        type="numpy" 
        # ▲ここにあった streaming=True を削除しました（これが重い原因）
    ),
    outputs=gr.Label(
        num_top_classes=3,
        label="リアルタイム判定結果"
    ),
    title="骨格推定じゃんけんAI",
    description="カメラに手をかざしてください。",
    live=True # これがあれば、自動的に連続判定してくれます
)

# 起動コマンド（サーバー設定を明確に指定）
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
