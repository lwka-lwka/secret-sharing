# 打包成 exe 用的啟動程式：點兩下後自動啟動 Streamlit，並在準備好時自動開啟瀏覽器
import os
import sys
import threading
import time
import urllib.request
import webbrowser

# 以下 import 讓 PyInstaller 把 image_sharing.py 會用到的套件一起打包
import numpy  # noqa: F401
import pandas  # noqa: F401
import PIL  # noqa: F401
import pyarrow  # noqa: F401
import streamlit  # noqa: F401
from streamlit.web import cli as stcli

PORT = 8502
URL = f"http://localhost:{PORT}"


def resource_path(name):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def open_browser_when_ready():
    # 等伺服器真的啟動後才開瀏覽器，避免開到空白頁
    for _ in range(300):
        try:
            with urllib.request.urlopen(URL + "/_stcore/health", timeout=1) as r:
                if r.status == 200:
                    webbrowser.open(URL)
                    return
        except Exception:
            pass
        time.sleep(1)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    print("正在啟動 (k, n)-threshold 秘密影像分享系統，請稍候，瀏覽器會自動開啟……")
    print(f"如果瀏覽器沒有自動開啟，請手動前往 {URL}")
    print("使用完畢後，關閉這個黑色視窗即可結束程式。")

    if os.environ.get("NO_BROWSER") != "1":
        threading.Thread(target=open_browser_when_ready, daemon=True).start()

    # server.headless=true：跳過 Streamlit 第一次執行時詢問 Email 的提示（否則程式會卡住）
    sys.argv = [
        "streamlit", "run", resource_path("image_sharing.py"),
        "--global.developmentMode=false",
        "--server.headless=true",
        f"--server.port={PORT}",
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(stcli.main())
