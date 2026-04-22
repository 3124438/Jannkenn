import streamlit as st
import streamlit.components.v1 as components
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
    st.error("⚠️ MediaPipeの読み込みでエラーが発生しました。")
    st.code(traceback.format_exc())
    st.stop()

# --- セッションステートの初期化 ---
if 'target_hand' not in st.session_state:
    st.session_state.target_hand = 'グー'

# --- ショートカットキー（JavaScript） ---
components.html(
    """
    <script>
    const doc = window.parent.document;
    doc.addEventListener('keydown', function(e) {
        if (e.code === 'Space') {
            e.preventDefault(); 
            const cameraBox = doc.querySelector('[data-testid="stCameraInput"]');
            if(cameraBox) {
                const btn = cameraBox.querySelector('button');
                if(btn) btn.click();
            }
        }
        if (e.code === 'KeyO') {
            const expanderBtn = doc.querySelector('[data-testid="stExpander"] summary');
            if(expanderBtn) expanderBtn.click();
        }
        function clickButtonByText(text) {
            const buttons = Array.from(doc.querySelectorAll('button'));
            const btn = buttons.find(b => b.innerText.includes(text));
            if(btn) btn.click();
        }
        if (e.code === 'KeyV') clickButtonByText('グー (V)');
        if (e.code === 'KeyB') clickButtonByText('チョキ (B)');
        if (e.code === 'KeyN') clickButtonByText('パー (N)');
    });
    </script>
    """,
    height=0,
    width=0,
)

# --- 設定 ---
CLASS_NAMES_JP = ['グー', 'チョキ', 'パー']
MODEL_FILENAME = 'my_janken_model.keras'

st.title("じゃんけんAI 🎯バウンディングボックス版")
st.write("目標を選んで撮影！AIの視点をのぞいてみよう。")

# --- 目標の選択UI ---
st.markdown("### 1. 目標の形を選ぶ")
col1, col2, col3 = st.columns(3)

if col1.button("✊ グー (V)", use_container_width=True):
    st.session_state.target_hand = 'グー'
if col2.button("✌️ チョキ (B)", use_container_width=True):
    st.session_state.target_hand = 'チョキ'
if col3.button("✋ パー (N)", use_container_width=True):
    st.session_state.target_hand = 'パー'

st.info(f"現在の目標: **【 {st.session_state.target_hand} 】**")

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
    # --- カメラ撮影 ---
    st.markdown("### 2. 撮影する")
    camera_image = st.camera_input("ここをクリックして撮影")

    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

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
                
                # --- リニアゲージ表示用の設定 ---
                max_index = np.argmax(scores)
                st.success(f"AIの判定結果: **{CLASS_NAMES_JP[max_index]}**")
                
                target_idx = CLASS_NAMES_JP.index(st.session_state.target_hand)
                target_score = scores[target_idx] * 100

                if target_score < 33:
                    status, pointer_color = "不明", "#7f8c8d"
                elif target_score < 60:
                    status, pointer_color = "迷走", "#2980b9"
                elif target_score < 80:
                    status, pointer_color = "懸念", "#d35400"
                else:
                    status, pointer_color = "確信", "#27ae60"

                st.markdown(f"### 🎯 目標【{st.session_state.target_hand}】との一致度")
                gauge_html = f"""
                <div style="font-family: sans-serif; padding-top: 40px; padding-bottom: 10px;">
                    <div style="position: relative; width: 98%; margin: 0 auto;">
                        <div style="position: absolute; left: {target_score}%; top: -35px; transform: translateX(-50%); text-align: center; z-index: 10;">
                            <div style="background-color: {pointer_color}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 14px; white-space: nowrap; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                                {status} {target_score:.1f}%
                            </div>
                            <div style="width: 0; height: 0; border-left: 8px solid transparent; border-right: 8px solid transparent; border-top: 10px solid {pointer_color}; margin: 0 auto;"></div>
                            <div style="width: 2px; height: 35px; background-color: {pointer_color}; margin: 0 auto;"></div>
                        </div>
                        <div style="display: flex; height: 30px; border-radius: 6px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);">
                            <div style="width: 33%; background-color: #bdc3c7; display: flex; align-items: center; justify-content: center; color: #2c3e50; font-weight: bold; font-size: 13px; border-right: 1px solid white;">不明</div>
                            <div style="width: 27%; background-color: #3498db; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 13px; border-right: 1px solid white;">迷走</div>
                            <div style="width: 20%; background-color: #f39c12; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 13px; border-right: 1px solid white;">懸念</div>
                            <div style="width: 20%; background-color: #2ecc71; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 13px;">確信</div>
                        </div>
                    </div>
                </div>
                """
                components.html(gauge_html, height=100)
                st.write("---")

                # 🌟 --- 詳細を見る：バウンディングボックスの描画 --- 🌟
                with st.expander("詳細を見る（AIの視点）"):
                    # 骨格とバウンディングボックスの描画準備
                    annotated_image = cv_img.copy()
                    h, w, _ = annotated_image.shape
                    
                    # 21個の点から最小・最大の座標を探す（枠の計算）
                    x_list = [lm.x for lm in hand_landmarks.landmark]
                    y_list = [lm.y for lm in hand_landmarks.landmark]
                    
                    # 座標をピクセルサイズに変換（少し余白を作る）
                    x_min, x_max = int(min(x_list) * w) - 20, int(max(x_list) * w) + 20
                    y_min, y_max = int(min(y_list) * h) - 20, int(max(y_list) * h) + 20
                    
                    # --- バウンディングボックスを描画 (緑色の四角) ---
                    cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                    # 枠の上に「HAND」とラベルを付ける
                    cv2.putText(annotated_image, "DETECTED HAND", (x_min, y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # --- 骨格（点と線）も重ねて描画 ---
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), 
                             caption="AIが『ここが手だ！』と判断したエリア（バウンディングボックス）")

                    # 予測確率の再表示
                    st.write("#### AIの各予測確率")
                    for i, name in enumerate(CLASS_NAMES_JP):
                        st.progress(float(scores[i]), text=f"{name}: {scores[i]*100:.1f}%")

                    # 座標データ表
                    st.write("--- 関節の座標データ ---")
                    coord_list = [{"関節番号": f"第{i}関節", "X": f"{data[i*2]:.4f}", "Y": f"{data[i*2+1]:.4f}"} for i in range(21)]
                    st.dataframe(pd.DataFrame(coord_list), use_container_width=True)

            else:
                st.warning("手が見つかりませんでした。もっとカメラに近づけてみてください。")
