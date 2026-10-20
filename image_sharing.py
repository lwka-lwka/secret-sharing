import io
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, PngImagePlugin
from streamlit import runtime

# 直接用 python 執行（例如 VS Code 的 ▶ 按鈕）時，自動改用 streamlit run 開啟網頁
if __name__ == "__main__" and not runtime.exists():
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    sys.exit()

st.set_page_config(page_title="(k, n)-threshold 秘密影像分享實作系統", layout="wide")

P = 251  # 小於 256 的最大質數，運算結果 0~250 可以直接存成 8-bit 像素
MARK = 250  # 像素值 >= 250 時的拆解標記


# --------------------------------------------------
# 1. 核心演算法
# --------------------------------------------------
def expand_stream(pixels):
    """無失真前處理：像素值 v >= 250 拆成兩個值 [250, v - 250]，其餘不變。
    拆完後每個值都在 0~250 之間，小於質數 p = 251。"""
    pixels = pixels.astype(np.int64)
    big = pixels >= MARK
    pos = np.arange(len(pixels)) + np.cumsum(big) - big  # 每個原像素在新序列中的位置
    out = np.zeros(len(pixels) + int(big.sum()), dtype=np.int64)
    out[pos] = np.where(big, MARK, pixels)
    out[pos[big] + 1] = pixels[big] - MARK
    return out


def shrink_stream(stream):
    """expand_stream 的反運算：遇到 250 就把下一個值加回去。
    標記後面的餘數只會是 0~5，不會是 250，所以每個 250 一定是標記。"""
    stream = np.asarray(stream, dtype=np.int64)
    is_mark = stream == MARK
    mark_pos = np.flatnonzero(is_mark)
    result = stream.copy()
    result[mark_pos] = MARK + stream[mark_pos + 1]
    keep = np.ones(len(stream), dtype=bool)
    keep[mark_pos + 1] = False
    return result[keep]


def share_mode_for(orig_mode):
    return "L" if orig_mode == "L" else "RGB"


def make_image_shares(img, k, n, xs=None, rng=None):
    """編碼端：把影像切成 n 張影像碎片。
    每個值 d 都當成一個秘密，建立 F(x) = (d + c1·x + … + c(k-1)·x^(k-1)) mod 251，
    第 i 張碎片存的是 F(xi)。"""
    rng = rng or np.random.default_rng()
    xs = list(xs) if xs is not None else list(range(1, n + 1))
    arr = np.asarray(img)
    secret = expand_stream(arr.reshape(-1))
    length = len(secret)

    coeffs = rng.integers(0, P, size=(k - 1, length), dtype=np.int64)  # 隨機係數 < p
    # 碎片影像：灰階原圖 → 灰階碎片；彩色原圖 → 彩色碎片（寬度與原圖相同，高度依需要的值數而定）
    share_mode = share_mode_for(img.mode)
    cs = 1 if share_mode == "L" else 3
    width = arr.shape[1]
    height = -(-length // (width * cs))  # 無條件進位

    shares = []
    for x in xs:
        y = secret.copy()
        for power in range(1, k):
            y = (y + coeffs[power - 1] * pow(x, power, P)) % P
        padded = np.zeros(width * height * cs, dtype=np.uint8)
        padded[:length] = y
        meta = {
            "ss_x": x, "ss_k": k, "ss_n": n, "ss_len": length,
            "orig_w": arr.shape[1], "orig_h": arr.shape[0], "orig_mode": img.mode,
        }
        shares.append({"x": x, "image": Image.fromarray(padded.reshape((height, width) + ((cs,) if cs > 1 else ())), share_mode), "meta": meta})
    return shares


def lagrange_weights(xs):
    """Lagrange 在 x = 0 的權重：Lj(0) = Π_{m≠j} (0 - xm) / (xj - xm) mod p。
    所有像素用的碎片相同，所以權重只要算一次。"""
    steps = []
    for j, xj in enumerate(xs):
        others = [xm for m, xm in enumerate(xs) if m != j]
        num, den = 1, 1
        for xm in others:
            num *= (0 - xm)
            den *= (xj - xm)
        num_mod, den_mod = num % P, den % P
        inv = pow(den_mod, -1, P)
        steps.append({
            "xj": xj, "others": others, "num": num, "den": den,
            "num_mod": num_mod, "den_mod": den_mod, "inv": inv,
            "weight": (num_mod * inv) % P,
        })
    return steps


def recover_image(shares):
    """解碼端：用蒐集到的碎片，對每個值做 Lagrange interpolation 求 F(0)。"""
    meta = shares[0]["meta"]
    length = meta["ss_len"]
    xs = [s["x"] for s in shares]
    steps = lagrange_weights(xs)
    secret = np.zeros(length, dtype=np.int64)
    for s, step in zip(shares, steps):
        y = np.asarray(s["image"], dtype=np.int64).reshape(-1)[:length]
        secret = (secret + y * step["weight"]) % P

    try:
        pixels = shrink_stream(secret)
    except IndexError:  # 碎片不足時，序列結構會亂掉
        pixels = secret
    channels = {"L": 1, "RGB": 3, "RGBA": 4}[meta["orig_mode"]]
    shape = (meta["orig_h"], meta["orig_w"]) + ((channels,) if channels > 1 else ())
    need = int(np.prod(shape))
    if len(pixels) != need:  # 長度不符（碎片不足）時補齊或截斷，只為了能顯示成影像
        pixels = np.resize(pixels, need)
    img = Image.fromarray(pixels.clip(0, 255).astype(np.uint8).reshape(shape), meta["orig_mode"])
    return img, steps, secret


def share_to_png_bytes(share):
    info = PngImagePlugin.PngInfo()
    for key, value in share["meta"].items():
        info.add_text(key, str(value))
    buf = io.BytesIO()
    share["image"].save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def share_from_upload(file):
    img = Image.open(file)
    text = getattr(img, "text", {}) or {}
    if "ss_x" not in text:
        raise ValueError(f"{file.name} 不是本系統產生的影像碎片（缺少碎片資訊）")
    meta = {key: (text[key] if key == "orig_mode" else int(text[key]))
            for key in ["ss_x", "ss_k", "ss_n", "ss_len", "orig_w", "orig_h", "orig_mode"]}
    return {"x": meta["ss_x"], "image": img.convert(share_mode_for(meta["orig_mode"])), "meta": meta}


def normalize_image(img):
    """統一轉成 L（灰階）、RGB 或 RGBA 三種模式"""
    if img.mode in ("L", "RGB", "RGBA"):
        return img.copy()
    if img.mode in ("1", "I;16", "I", "F"):
        return img.convert("L")
    if "A" in img.getbands() or "transparency" in img.info:
        return img.convert("RGBA")
    return img.convert("RGB")


def sample_image():
    """內建範例影像：彩色漸層背景 + 文字"""
    from PIL import ImageDraw, ImageFont
    w, h = 320, 200
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.stack([(xx * 255 // w), (yy * 255 // h), np.full_like(xx, 180)], axis=-1).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=44)
    except TypeError:
        font = ImageFont.load_default()
    draw.rectangle([20, 55, 300, 145], fill=(255, 255, 255))
    draw.text((40, 70), "SECRET 2027", fill=(200, 0, 0), font=font)
    return img


def png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# --------------------------------------------------
# 2. 介面
# --------------------------------------------------
st.title("(k, n)-threshold 秘密影像分享實作系統")
st.caption("把一張機密影像切成 n 張影像碎片，任意 k 張即可完整還原，少於 k 張只會得到雜訊。"
           f"所有運算都在 mod p = {P} 下進行。")

# ================= 編碼端 =================
with st.container(border=True):
    st.subheader("編碼端")

    use_sample = st.toggle("使用內建範例影像", value=False)
    uploaded = None
    if not use_sample:
        uploaded = st.file_uploader("1. 輸入一張機密影像（PNG / BMP / JPG）", type=["png", "bmp", "jpg", "jpeg"])

    col1, col2 = st.columns(2)
    with col1:
        k = int(st.number_input("門檻值 k", min_value=2, max_value=10, value=3, step=1))
    with col2:
        n = int(st.number_input("碎片數 n（n ≥ k）", min_value=2, max_value=10, value=5, step=1))

    if use_sample:
        secret_img, src_name = sample_image(), "範例影像"
    elif uploaded is not None:
        secret_img, src_name = normalize_image(Image.open(uploaded)), uploaded.name
    else:
        secret_img, src_name = None, None

    if n < k:
        st.error("n 必須大於等於 k")
    split_btn = st.button("🔑 2. 切分成 n 張影像碎片", type="primary", use_container_width=True,
                          disabled=secret_img is None or n < k)

    if split_btn:
        shares = make_image_shares(secret_img, k, n)
        st.session_state.pop("rec_sel", None)
        st.session_state.img_ss = {
            "k": k, "n": n, "src": src_name, "original": secret_img,
            "shares": shares, "png": [share_to_png_bytes(s) for s in shares],
        }

    ss = st.session_state.get("img_ss")
    if ss and (ss["k"], ss["n"], ss["src"]) != (k, n, src_name):
        st.info("參數或影像已變更，請重新按「切分成 n 張影像碎片」。")
        ss = None

    if ss:
        orig = ss["original"]
        st.markdown(f"**原始機密影像**（{orig.width} × {orig.height}，{orig.mode}）")
        st.image(orig, width=320)

        length = ss["shares"][0]["meta"]["ss_len"]
        total_px = orig.width * orig.height * len(orig.getbands())
        st.markdown(f"**所有影像碎片**（共 {ss['n']} 張，每張 {ss['shares'][0]['image'].width} × "
                    f"{ss['shares'][0]['image'].height}，{ss['shares'][0]['image'].mode}）")
        st.caption(f"原影像共 {total_px} 個像素值，其中 {length - total_px} 個 ≥ 250 的值被拆成兩個，"
                   f"所以每張碎片存 {length} 個值。")
        per_row = 5
        for row_start in range(0, ss["n"], per_row):
            cols = st.columns(per_row)
            for i in range(row_start, min(row_start + per_row, ss["n"])):
                s = ss["shares"][i]
                with cols[i - row_start]:
                    st.image(s["image"], caption=f"q{i + 1}（x = {s['x']}）", use_container_width=True)
                    st.download_button("下載", ss["png"][i], file_name=f"share_q{i + 1}.png",
                                       mime="image/png", key=f"dl_{i}", use_container_width=True)

        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w") as zf:
            for i, data in enumerate(ss["png"]):
                zf.writestr(f"share_q{i + 1}.png", data)
        st.download_button("📦 下載全部碎片（zip）", zbuf.getvalue(), file_name="image_shares.zip",
                           mime="application/zip", use_container_width=True)

st.write("")

# ================= 解碼端 =================
with st.container(border=True):
    st.subheader("解碼端")

    source = st.radio("碎片來源", ["使用上方產生的碎片", "上傳碎片檔案（PNG）"], horizontal=True)

    picked, labels = [], []
    if source == "使用上方產生的碎片":
        if not ss:
            st.info("💡 請先在編碼端按「切分成 n 張影像碎片」，或改用上傳碎片檔案。")
        else:
            all_labels = [f"q{i}" for i in range(1, ss["n"] + 1)]
            labels = st.multiselect(f"模擬蒐集到的碎片（至少 {ss['k']} 張才能還原）", all_labels,
                                    default=all_labels[:ss["k"]])
            labels = sorted(labels, key=all_labels.index)
            picked = [ss["shares"][all_labels.index(c)] for c in labels]
    else:
        files = st.file_uploader("上傳影像碎片（可一次選多張）", type=["png"], accept_multiple_files=True)
        for f in files or []:
            try:
                picked.append(share_from_upload(f))
                labels.append(f.name)
            except Exception as e:
                st.error(str(e))
        if picked:
            base = picked[0]["meta"]
            if any(p["meta"]["ss_len"] != base["ss_len"] or p["meta"]["orig_w"] != base["orig_w"] for p in picked):
                st.error("這些碎片不是來自同一張影像")
                picked = []
            elif len({p["x"] for p in picked}) != len(picked):
                st.error("有重複的碎片（x 值相同）")
                picked = []

    sel_key = tuple(labels)
    if st.button("▶ 執行還原", type="primary", use_container_width=True, disabled=not picked):
        st.session_state.rec_sel = sel_key

    if picked and st.session_state.get("rec_sel") == sel_key:
        need_k = picked[0]["meta"]["ss_k"]
        rec_img, steps, secret = recover_image(picked)

        st.markdown("#### 計算過程")

        # 步驟 1：Lagrange 權重
        st.markdown(f"**步驟 1：計算每張碎片的 Lagrange 權重 Lj(0)**（mod {P}）")
        st.caption("所有像素用的是同一組碎片（同一組 x），所以權重只要算一次，就能套用到每個像素。")
        for j, stp in enumerate(steps):
            num_expr = "".join(f"(0 − {xm})" for xm in stp["others"]) or "1"
            den_expr = "".join(f"({stp['xj']} − {xm})" for xm in stp["others"]) or "1"
            with st.container(border=True):
                st.markdown(
                    f"**{labels[j]}（x = {stp['xj']}）**\n\n"
                    f"- 分子 = {num_expr} = {stp['num']} ≡ **{stp['num_mod']}** (mod {P})\n"
                    f"- 分母 = {den_expr} = {stp['den']} ≡ **{stp['den_mod']}** (mod {P})\n"
                    f"- 分母的模反元素：{stp['den_mod']}⁻¹ ≡ **{stp['inv']}** (mod {P})"
                    f"　（驗算：{stp['den_mod']} × {stp['inv']} = {stp['den_mod'] * stp['inv']} ≡ 1）\n"
                    f"- 權重 L(0) = 分子 × 分母⁻¹ = {stp['num_mod']} × {stp['inv']} ≡ **{stp['weight']}** (mod {P})"
                )

        # 步驟 2：逐像素還原（列出前幾個當例子）
        st.markdown("**步驟 2：每個像素位置 d = Σ F(xj) × Lj(0) mod 251**（以下列出前 8 個位置為例）")
        rows = []
        ys = [np.asarray(p["image"], dtype=np.int64).reshape(-1) for p in picked]
        for pos in range(min(8, len(secret))):
            terms = " + ".join(f"{int(y[pos])}×{stp['weight']}" for y, stp in zip(ys, steps))
            total = sum(int(y[pos]) * stp["weight"] for y, stp in zip(ys, steps))
            rows.append({"位置": pos, "計算": f"({terms}) mod {P}", "= ": f"{total} mod {P}", "d": int(secret[pos])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # 步驟 3：還原 250 標記
        st.markdown("**步驟 3：把拆開的值合併回來，排回原影像大小**")
        st.caption("編碼時 ≥ 250 的像素被拆成 [250, v − 250]，所以還原時遇到 250，就把下一個值加回去。")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**還原結果**")
            st.image(rec_img, width=320)
            st.download_button("下載還原影像", png_bytes(rec_img), file_name="recovered.png", mime="image/png")
        with col_b:
            if source == "使用上方產生的碎片" and ss:
                st.markdown("**原始機密影像**")
                st.image(ss["original"], width=320)

        if len(picked) >= need_k:
            if source == "使用上方產生的碎片" and ss:
                same = np.array_equal(np.asarray(rec_img), np.asarray(ss["original"]))
                if same:
                    st.success(f"使用 {len(picked)} 張碎片還原，與原始影像每個像素完全相同 ✅（無失真）")
                else:
                    st.error("還原結果與原始影像不同 ❌")
            else:
                st.success(f"使用 {len(picked)} 張碎片（≥ k = {need_k}）完成還原。")
        else:
            st.warning(f"只有 {len(picked)} 張碎片（少於 k = {need_k}），還原出來只是雜訊——碎片不足時無法得知原影像。")
    elif picked:
        st.caption("選好碎片後，按「執行還原」查看計算過程與還原結果。")
