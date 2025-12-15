import os
import time
import requests
import streamlit as st
from PIL import Image
from io import BytesIO

# =====================================
# 🔐 Hugging Face APIキーの安全な読み込み
# =====================================
def get_hf_token() -> str:
    # 1) Streamlit secrets（推奨）
    if "HF_TOKEN" in st.secrets:
        return st.secrets["HF_TOKEN"]

    # 2) 環境変数
    env = os.getenv("HF_TOKEN")
    if env:
        return env

    return ""

HF_TOKEN = get_hf_token()

if not HF_TOKEN:
    st.error("HF_TOKEN が見つかりません。`.streamlit/secrets.toml` か環境変数 `HF_TOKEN` を設定してください。")
    st.stop()

# =====================================
# ✅ 画像生成モデル（Stable Diffusion）
# =====================================
# SDXL（高品質、やや重い）
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# 503（モデル読み込み中）対策：少し待ってリトライする
def call_hf_text2image(prompt: str, negative_prompt: str = "", steps: int = 30, guidance: float = 7.0,
                      width: int = 1024, height: int = 1024, seed: int | None = None,
                      max_retries: int = 3):
    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height,
        },
        "options": {
            # 503 "Model is currently loading" を減らす（それでも起きる時はある）
            "wait_for_model": True
        }
    }

    if negative_prompt:
        payload["parameters"]["negative_prompt"] = negative_prompt
    if seed is not None:
        payload["parameters"]["seed"] = seed

    for attempt in range(max_retries):
        r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=180)

        # 正常：画像バイトが返ることが多い（content-type image/*）
        ctype = r.headers.get("content-type", "")
        if r.status_code == 200 and ctype.startswith("image/"):
            return r.content

        # 503: model loading など（JSONで返るケース多い）
        if r.status_code in (503, 504):
            try:
                j = r.json()
                est = j.get("estimated_time", 10)
            except Exception:
                est = 10
            time.sleep(min(max(est, 3), 30))
            continue

        # 200でもJSONが返る時がある（エラー等）
        try:
            err = r.json()
        except Exception:
            err = r.text

        raise RuntimeError(f"HF API error: status={r.status_code}, content-type={ctype}, body={err}")

    raise RuntimeError("モデルが混雑/読み込み中で失敗しました。時間をおいて再実行してください。")

# =====================================
# UI
# =====================================
st.title("🎨 画像生成（Stable Diffusion / Hugging Face API）")

st.caption("※無料枠は混雑すると 503 が出ることがあります（モデル読み込み中）。その場合は少し待って再実行してください。")

with st.sidebar:
    st.markdown("## 📚 明治時代 学習ページ")
    st.markdown(
        "- [明治文化学習（Google Sites）]"
        "(https://sites.google.com/view/meijibunkagakusyuu/ホーム)"
    )

MEIJI_BOOST = """
Japan Meiji era (1868-1912), Meiji period, historical streetscape,
brick western-style architecture, gas lamps, rickshaw,
kimono and western clothing (yofuku), early modern Japan,
highly detailed, cinematic lighting
""".strip()

user_prompt = st.text_input("作りたい絵の説明（日本語OK）", "煉瓦造りの洋風建築が立ち並ぶ街並み")
negative = st.text_input("絵に入れたくない要素（任意）", "low quality, blurry, deformed, extra fingers")

final_prompt = f"{user_prompt}, {MEIJI_BOOST}"

with st.expander("生成パラメータ（任意）"):
    steps = st.slider("試行回数(高いほど高クオリティ)", 10, 50, 30)
    guidance = st.slider("指示文にどれだけ従わせるか", 1.0, 12.0, 7.0)
    size = st.selectbox("サイズ", ["1024x1024", "768x768", "512x512"], index=0)
    seed_text = st.text_input("Seed（空ならランダム）", "")

w, h = map(int, size.split("x"))
seed = int(seed_text) if seed_text.strip().isdigit() else None

if st.button("画像を生成する ✨"):
    with st.spinner("生成中..."):
        try:
            img_bytes = call_hf_text2image(
                prompt=final_prompt,
                negative_prompt=negative,
                steps=steps,
                guidance=guidance,
                width=w,
                height=h,
                seed=seed,
                max_retries=4
            )
            image = Image.open(BytesIO(img_bytes))
            st.image(image, caption=f"Model: {MODEL_ID}", use_container_width=True)

            # ダウンロード
            buf = BytesIO()
            image.save(buf, format="PNG")
            st.download_button(
                label="PNGをダウンロード",
                data=buf.getvalue(),
                file_name="generated.png",
                mime="image/png",
            )

        except Exception as e:
            st.error(f"画像生成エラー: {e}")
