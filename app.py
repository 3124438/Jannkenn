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
    st.error("⚠️ MediaPipeの読み込みでエラーが発生しました。以下のログをコピーして教えてください。")
    st.code(traceback.format_exc())
    st.stop()

# --- セッションステートの初期化（目標の手を保存） ---
if 'target_hand' not in st.session_state:
    st.session_state.target_hand = 'グー'

# --- ショートカットキー用の裏技スクリプト（JavaScript） ---
components.html(
    """
    <script>
    const doc = window.parent.document;
    doc.addEventListener('keydown', function(e) {
        // スペースキーが押された時の処理（撮影）
        if (e.code === 'Space') {
            e.preventDefault(); 
            const cameraBox = doc.querySelector('[data-testid="stCameraInput"]');
            if(cameraBox) {
                const btn = cameraBox.querySelector('button');
                if(btn) btn.click();
            }
        }
        // O(オー)キーが押された時の処理（種明かし展開）
        if (e.code === 'KeyO') {
            const expanderBtn = doc.querySelector('[data-testid="stExpander"] summary');
            if(expanderBtn) expanderBtn.click();
        }
        // V, B, Nキーで目標を選択する処理
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

st.title("じゃんけんAI 🎯乖離チェック版")
st.write("先に作りたい形を選んでから、Webカメラで撮影してください！")

# --- 目標の選択UI ---
st.markdown("### 1. 目標の形を選ぶ")
col1, col2, col3 = st.columns(3)

if col1.button("✊ グー (V)", use_container_width=True):
    st.session_state.target_hand = 'グー'
if col2.button("✌️ チョキ (B)", use_container_width=True):
    st.session_state.target_hand = 'チョキ'
if col3.button("✋ パー (N)", use_container_width=True):
    st.session_state.target_hand = 'パー'

st.info(f"現在の目標: **【 {st.session_state.target_hand} 】** （ショートカット：スペースキーで撮影）")

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
    # --- カメラウィジェット（写真撮影モード） ---
    st.markdown("### 2. 撮影する")
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
                
                # --- 結果の表示 ---
                max_index = np.argmax(scores)
                st.success(f"AIの判定結果: **{CLASS_NAMES_JP[max_index]}**")
                
                # --- 目標に対するスコア計算とグラフ表示 ---
                target_idx = CLASS_NAMES_JP.index(st.session_state.target_hand)
                target_score = scores[target_idx] * 100

                # スコアに応じた判定と色の設定
                if target_score < 33:
                    status = "不明"
                    bar_color = "#808080" # グレー
                elif target_score < 60:
                    status = "迷走"
                    bar_color = "#3498db" # 青
                elif target_score < 80:
                    status = "懸念"
                    bar_color = "#f39c12" # オレンジ
                else:
                    status = "確信"
                    bar_color = "#2ecc71" # 緑

                st.markdown(f"### 🎯 目標【{st.session_state.target_hand}】との一致度")
                
                # カスタムHTML/CSSでスコアバー（ゲージ）を描画
                st.markdown(f"""
                <div style="background-color: #e6e6e6; border-radius: 10px; width: 100%; height: 30px; margin-top: 10px; position: relative; overflow: hidden;">
                    <div style="background-color: {bar_color}; width: {target_score}%; height: 100%; border-radius: 10px; transition: width 0.5s ease-in-out;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
                    <span style="font-size: 24px; font-weight: bold; color: {bar_color};">{target_score:.1f}%</span>
                    <span style="font-size: 20px; font-weight: bold; background-color: {bar_color}; color: white; padding: 4px 12px; border-radius: 15px;">{status}</span>
                </div>
                <hr>
                """, unsafe_allow_html=True)

                # 🌟 --- 種明かし機能（詳細情報） --- 🌟
                with st.expander("詳細を見る"):
                    # 以前の外出しされていた各確率バーをここに移動
                    st.write("#### AIの各予測確率")
                    for i, name in enumerate(CLASS_NAMES_JP):
                        st.progress(float(scores[i]), text=f"{name}: {scores[i]*100:.1f}%")

                    st.write("---")
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
