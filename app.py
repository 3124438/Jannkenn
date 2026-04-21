import streamlit as st
import streamlit.components.v1 as components # ショートカットの裏技用に使いします
import tensorflow as tf
import numpy as np
import cv2
import traceback
import pandas as pd

# --- MediaPipeの安全な読み込み ---
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except Exception as e:
    st.error("⚠️ MediaPipeの読み込みでエラーが発生しました。以下のログをコピーして教えてください。")
    st.code(traceback.format_exc())
    st.stop()

# --- ショートカットキー用の裏技スクリプト（JavaScript） ---
# スペースキーでカメラボタンを押し、Oキーで種明かしメニューを開閉します
components.html(
    """
    <script>
    const doc = window.parent.document;
    doc.addEventListener('keydown', function(e) {
        // スペースキーが押された時の処理
        if (e.code === 'Space') {
            e.preventDefault(); // スペースキーによる画面スクロールを防止
            const cameraBox = doc.querySelector('[data-testid="stCameraInput"]');
            if(cameraBox) {
                const btn = cameraBox.querySelector('button');
                if(btn) btn.click(); // カメラの「撮影」「クリア」ボタンを自動クリック
            }
        }
        // O(オー)キーが押された時の処理
        if (e.code === 'KeyO') {
            const expanderBtn = doc.querySelector('[data-testid="stExpander"] summary');
            if(expanderBtn) expanderBtn.click(); // 種明かしメニューを自動クリック
        }
    });
    </script>
    """,
    height=0,
    width=0,
)

# --- 設定 ---
CLASS_NAMES_JP = ['グー', 'チョキ', 'パー']
MODEL_FILENAME = 'my_janken_model.keras'

st.title("じゃんけんAI")
st.write("Webカメラで手を撮影してください！")
st.info("⌨️ **操作ショートカット** : `スペースキー` 撮影/削除 ｜ `O(オー)キー` 骨格データの表示切替")

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
            static_image_mode=True, 
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

                # 🌟 --- 種明かし機能 --- 🌟
                st.write("---")
                with st.expander("AIが見ている「骨格データ」を見る"):
                    st.write("AIは画像の見た目ではなく、手の21個の関節の **(X, Y)座標**（42個の数字）を読み取って判定しています！")
                    
                    # 1. 骨格を描画した画像を表示
                    annotated_image = cv_img.copy()
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="AIが認識した骨格")

                    # 2. 実際の座標データを表にして表示
                    coord_list = []
                    for i in range(21):
                        coord_list.append({
                            "関節番号": f"第{i}関節",
                            "X座標 (横)": f"{data[i*2]:.4f}",
                            "Y座標 (縦)": f"{data[i*2+1]:.4f}"
                        })
                    st.dataframe(pd.DataFrame(coord_list), use_container_width=True)

            else:
                st.warning("手が見つかりませんでした。もう一度、手をカメラの中央に映して撮影してください。")
