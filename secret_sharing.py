import random
import subprocess
import sys

import pandas as pd
import streamlit as st
from streamlit import runtime

# 直接用 python 執行（例如 VS Code 的 ▶ 按鈕）時，自動改用 streamlit run 開啟網頁
if __name__ == "__main__" and not runtime.exists():
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    sys.exit()

st.set_page_config(page_title="(k, n)-threshold 秘密分享實作系統", layout="wide")


# --------------------------------------------------
# 1. 核心演算法
# --------------------------------------------------
def is_prime(m):
    if m < 2:
        return False
    if m % 2 == 0:
        return m == 2
    i = 3
    while i * i <= m:
        if m % i == 0:
            return False
        i += 2
    return True


def next_prime(m):
    m = max(m, 2)
    while not is_prime(m):
        m += 1
    return m


def poly_eval(d, coeffs, x, p):
    """F(x) = (d + c1*x + c2*x^2 + ... + c(k-1)*x^(k-1)) mod p"""
    result = d
    for power, c in enumerate(coeffs, start=1):
        result += c * pow(x, power)
    return result % p


def make_shares(d, k, n, p, coeffs, xs):
    """編碼端：把秘密 d 切成 n 片碎片 (xi, F(xi))"""
    return [(x, poly_eval(d, coeffs, x, p)) for x in xs]


def lagrange_recover(shares, p):
    """解碼端：用 Lagrange interpolation 在 x = 0 求回 d
    d = Σ F(xj) * Π_{m≠j} (0 - xm) / (xj - xm)   (mod p)
    """
    d = 0
    for j, (xj, yj) in enumerate(shares):
        num, den = 1, 1
        for m, (xm, _) in enumerate(shares):
            if m == j:
                continue
            num = (num * (-xm)) % p
            den = (den * (xj - xm)) % p
        term = yj * num * pow(den, -1, p)  # pow(den, -1, p) = den 的模反元素
        d = (d + term) % p
    return d


def poly_text(d, coeffs, p):
    terms = [str(d)]
    for power, c in enumerate(coeffs, start=1):
        terms.append(f"{c}x" if power == 1 else f"{c}x^{power}")
    return f"F(x) = ({' + '.join(terms)}) mod {p}"


# --------------------------------------------------
# 2. 介面
# --------------------------------------------------
st.title("(k, n)-threshold 秘密分享實作系統")
st.caption("Shamir Secret Sharing：把一把 key 切成 n 片碎片，任意 k 片即可還原，少於 k 片則無法得知 key。")

use_example = st.toggle("套用課本範例：(3, 5)-threshold，F(x) = (66 + 10x + 30x²) mod 67", value=False)

# ================= 編碼端 =================
with st.container(border=True):
    st.subheader("編碼端")

    col1, col2, col3 = st.columns(3)
    with col1:
        d = st.number_input("1. 輸入一把 key（秘密值 d，非負整數）", min_value=0,
                            value=66 if use_example else 2027, step=1, disabled=use_example)
    with col2:
        k = st.number_input("門檻值 k", min_value=2, value=3, step=1, disabled=use_example)
    with col3:
        n = st.number_input("碎片數 n（n ≥ k）", min_value=2, value=5, step=1, disabled=use_example)

    if use_example:
        d, k, n = 66, 3, 5

    d, k, n = int(d), int(k), int(n)
    if n < k:
        st.error("n 必須大於等於 k")
        st.stop()

    # 質數 p 必須 > d（才能還原 d）且 > n（才有 n 個相異且非 0 的 xi）
    p_min = next_prime(max(d, n) + 1)
    col4, col5 = st.columns(2)
    with col4:
        p = st.number_input(f"質數 p（需為質數且 > d、> n，建議值 {p_min}）", min_value=2,
                            value=67 if use_example else p_min, step=1, disabled=use_example)
        p = 67 if use_example else int(p)
    with col5:
        x_mode = st.radio("碎片的 xi 取法", ["1, 2, …, n", "隨機相異值"], horizontal=True, disabled=use_example)

    if not is_prime(p):
        st.error(f"{p} 不是質數")
        st.stop()
    if p <= d:
        st.error(f"p 必須大於 key（d = {d}），否則 d mod p 會遺失資訊")
        st.stop()
    if p <= n:
        st.error("p 必須大於 n，才能取出 n 個相異且不為 0 的 xi")
        st.stop()

    split_btn = st.button("🔑 2. 切分成 n 片碎片", type="primary", use_container_width=True)

    if split_btn:
        if use_example:
            coeffs = [10, 30]
            xs = [1, 5, 2, 11, 7]
        else:
            # 隨機係數 c1 ~ c(k-1) < p；最高次項不取 0，確保真的是 k-1 次多項式
            coeffs = [random.randrange(0, p) for _ in range(k - 2)] + [random.randrange(1, p)]
            if x_mode == "1, 2, …, n":
                xs = list(range(1, n + 1))
            else:
                xs = random.sample(range(1, p), n)
        st.session_state.pop("rec_sel", None)  # 重新切分後，清掉舊的還原結果
        st.session_state.ss = {
            "d": d, "k": k, "n": n, "p": p, "coeffs": coeffs,
            "shares": make_shares(d, k, n, p, coeffs, xs),
        }

    ss = st.session_state.get("ss")
    if ss and (ss["d"], ss["k"], ss["n"], ss["p"]) != (d, k, n, p):
        st.info("參數已變更，請重新按「切分成 n 片碎片」。")
        ss = None

    if ss:
        st.markdown("**多項式**")
        st.code(poly_text(ss["d"], ss["coeffs"], ss["p"]), language=None)
        st.markdown(f"**所有碎片之值**（共 {ss['n']} 片）")
        df_shares = pd.DataFrame({
            "碎片": [f"q{i}" for i in range(1, ss["n"] + 1)],
            "xi": [x for x, _ in ss["shares"]],
            "qi = F(xi)": [y for _, y in ss["shares"]],
        })
        st.dataframe(df_shares, hide_index=True, use_container_width=True)

st.write("")

# ================= 解碼端 =================
with st.container(border=True):
    st.subheader("解碼端")

    if not ss:
        st.info("💡 請先在編碼端按「切分成 n 片碎片」。")
        st.stop()

    k, n, p, shares = ss["k"], ss["n"], ss["p"], ss["shares"]
    labels = [f"q{i}" for i in range(1, n + 1)]

    st.markdown("**自行挑選碎片還原**")
    chosen = st.multiselect(f"模擬蒐集到的碎片（至少 {k} 片才能還原）", labels, default=labels[:k])
    chosen = sorted(chosen, key=labels.index)

    if st.button("▶ 執行還原", type="primary", use_container_width=True, disabled=not chosen):
        st.session_state.rec_sel = tuple(chosen)

    # 只有按下「執行還原」且選擇沒有再變動時，才顯示結果
    if st.session_state.get("rec_sel") == tuple(chosen) and chosen:
        picked = [shares[labels.index(c)] for c in chosen]
        xs = [x for x, _ in picked]

        st.markdown("#### 計算過程")

        # 步驟 1：逐項計算
        st.markdown("**步驟 1：逐項計算**")
        terms = []
        for j, (xj, yj) in enumerate(picked):
            others = [xm for m, xm in enumerate(xs) if m != j]
            num, den = 1, 1
            for xm in others:
                num *= (0 - xm)
                den *= (xj - xm)
            num_mod = num % p
            den_mod = den % p
            inv = pow(den_mod, -1, p)
            term = (yj * num_mod * inv) % p
            terms.append(term)

            num_expr = "".join(f"(0 − {xm})" for xm in others) or "1"
            den_expr = "".join(f"({xj} − {xm})" for xm in others) or "1"
            with st.container(border=True):
                st.markdown(
                    f"**第 {j + 1} 項：{chosen[j]} = ({xj}, {yj})**\n\n"
                    f"- 分子 = {num_expr} = {num} ≡ **{num_mod}** (mod {p})\n"
                    f"- 分母 = {den_expr} = {den} ≡ **{den_mod}** (mod {p})\n"
                    f"- 分母的模反元素：{den_mod}⁻¹ ≡ **{inv}** (mod {p})"
                    f"　（驗算：{den_mod} × {inv} = {den_mod * inv} ≡ 1）\n"
                    f"- 本項 = F({xj}) × 分子 × 分母⁻¹ = {yj} × {num_mod} × {inv} = {yj * num_mod * inv}"
                    f" ≡ **{term}** (mod {p})"
                )

        # 步驟 2：加總
        rec = sum(terms) % p
        st.markdown("**步驟 2：各項加總後取 mod p**")
        st.markdown(f"d = ({' + '.join(map(str, terms))}) mod {p} = {sum(terms)} mod {p} = **{rec}**")

        # 結果
        if len(chosen) >= k:
            if rec == ss["d"]:
                st.success(f"使用 {len(chosen)} 片碎片，算回 key = {rec}，與原本的 key 相同 ✅")
            else:
                st.error(f"算回 {rec}，與原本的 key（{ss['d']}）不同 ❌")
        else:
            st.warning(f"只有 {len(chosen)} 片碎片（少於 k = {k}），算出的 {rec} "
                       f"{'剛好與 key 相同，但只是巧合' if rec == ss['d'] else '不是原本的 key'}"
                       f"——碎片不足時無法還原 key。")
    elif chosen:
        st.caption("選好碎片後，按「執行還原」查看計算過程。")
