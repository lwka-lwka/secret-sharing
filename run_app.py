# 打包成 exe 用的啟動程式：執行後會自動用 Streamlit 開啟 secret_sharing.py 的網頁介面
import os
import sys

# 以下 import 讓 PyInstaller 把 secret_sharing.py 會用到的套件一起打包
import pandas  # noqa: F401
import pyarrow  # noqa: F401
import streamlit  # noqa: F401
from streamlit.web import cli as stcli


def resource_path(name):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


if __name__ == "__main__":
    sys.argv = [
        "streamlit", "run", resource_path("secret_sharing.py"),
        "--global.developmentMode=false",
        "--server.headless=false",
    ]
    sys.exit(stcli.main())
