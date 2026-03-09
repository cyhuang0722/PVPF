"""
简洁示例：上传路径文件后，通过 WebSocket 指定雷达使用该路径。
修改 RADAR_IP、PORT、PATH_JSON 即可运行。
"""

import json
from pathlib import Path

import requests
from websocket import create_connection

# ===== 根据实际情况修改 =====
RADAR_IP = "124.222.126.90"    # 或域名，例如 192.168.0.1
PORT = 9040
PATH_JSON = "path_20260220-114500.json"  # 待上传的路径文件
# ============================


def main() -> None:
    path = Path(PATH_JSON)
    if not path.is_file():
        raise FileNotFoundError(f"路径文件不存在: {path}")

    # 1) 上传路径文件（POST /upload）
    upload_url = f"http://{RADAR_IP}:{PORT}/upload"
    with path.open("rb") as f:
        res = requests.post(upload_url, files={"file": f}, timeout=10)
    print("upload status:", res.status_code, res.text)
    res.raise_for_status()

    # 2) 建立 WebSocket 连接
    ws_url = f"ws://{RADAR_IP}:{PORT}/ws"
    ws = create_connection(ws_url, timeout=5)
    print("websocket connected")

    # 3) 指定路径文件（field49）
    msg = {"field": "field49", "value": path.name}
    ws.send(json.dumps(msg))
    print("path selected:", path.name)

    # 4) 可选：读取一次回复
    try:
        reply = ws.recv()
        print("radar reply:", reply)
    except Exception:
        pass
    ws.close()


if __name__ == "__main__":
    main()