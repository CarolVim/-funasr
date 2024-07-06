# -*- encoding: utf-8 -*-
import os
import ssl
import asyncio
import argparse
import json
import pyaudio
import websockets
import webrtcvad
from queue import Queue
from llama_index.llms.ollama import Ollama

all_output = []

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--host",
                    type=str,
                    default="localhost",
                    required=False,
                    help="host ip, localhost, 0.0.0.0")
parser.add_argument("--port",
                    type=int,
                    default=10096,
                    required=False,
                    help="grpc server port")
parser.add_argument("--chunk_size",
                    type=str,
                    default="5, 10, 5",
                    help="chunk")
parser.add_argument("--chunk_interval",
                    type=int,
                    default=10,
                    help="chunk interval in ms")
parser.add_argument("--hotword",
                    type=str,
                    default="",
                    help="hotword file path, one hotword per line (e.g.: 阿里巴巴 20)")
parser.add_argument("--audio_fs",
                    type=int,
                    default=16000,
                    help="audio sample rate")
parser.add_argument("--use_itn",
                    type=int,
                    default=1,
                    help="1 for using ITN, 0 for not using ITN")
parser.add_argument("--mode",
                    type=str,
                    default="2pass",
                    help="offline, online, 2pass")
parser.add_argument("--ssl",
                    type=int,
                    default=1,
                    help="1 for SSL connect, 0 for no SSL")
parser.add_argument("--record_time",
                    type=int,
                    default=10,
                    help="Recording time in seconds")

args = parser.parse_args()
args.chunk_size = [int(x) for x in args.chunk_size.split(",")]

# 全局变量
voices = Queue()
offline_msg_done = False
recognized_text = ""
stop_signal = False

# 录音并发送音频数据到 WebSocket 服务器
async def record_microphone(websocket):
    global recognized_text, stop_signal
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = args.audio_fs
    CHUNK = int(RATE / 1000 * args.chunk_interval)

    vad = webrtcvad.Vad()
    vad.set_mode(1)

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # 读取热词文件并构建热词字典
    fst_dict = {}
    hotword_msg = ""
    if args.hotword.strip() != "":
        with open(args.hotword) as f_scp:
            hot_lines = f_scp.readlines()
            for line in hot_lines:
                words = line.strip().split(" ")
                if len(words) < 2:
                    print("Please check format of hotwords")
                    continue
                try:
                    fst_dict[" ".join(words[:-1])] = int(words[-1])
                except ValueError:
                    print("Please check format of hotwords")
            hotword_msg = json.dumps(fst_dict)

    use_itn = True
    if args.use_itn == 0:
        use_itn = False

    # 发送初始配置信息
    message = json.dumps({"mode": args.mode, "chunk_size": args.chunk_size, "chunk_interval": args.chunk_interval,
                          "wav_name": "microphone", "is_speaking": True, "hotwords": hotword_msg, "itn": use_itn})
    await websocket.send(message)

    silence_threshold = 20  # 阈值为连续20个静音帧（约0.4秒）
    silence_count = 0

    while True:
        data = stream.read(CHUNK)
        is_speech = vad.is_speech(data, RATE)
        if is_speech:
            silence_count = 0
            await websocket.send(data)
        else:
            silence_count += 1
            if silence_count > silence_threshold:
                break
        await asyncio.sleep(0.005)

    # 录音结束后，发送结束标志
    await websocket.send(json.dumps({"is_speaking": False}))
    stop_signal = True

    stream.stop_stream()
    stream.close()
    p.terminate()

# 接收服务器返回的消息并更新识别文本
async def message(websocket):
    global recognized_text, stop_signal
    try:
        while True:
            msg = await websocket.recv()
            # print(f"Received message: {msg}")  # Debugging output
            msg = json.loads(msg)
            text = msg["text"]
            recognized_text = text[-10000:]  # 只保留最新的 10000 字符
            if stop_signal:
                break
    except Exception as e:
        print("Exception:", e)

    # 完整识别后输出结果
    os.system('clear')
    print("\rRecognized Text: " + recognized_text, end='', flush=True)
    print("\n语音识别完成")

# WebSocket 客户端，连接服务器并启动录音和消息处理任务
async def ws_client():
    global websocket
    if args.ssl == 1:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        uri = f"wss://{args.host}:{args.port}"
    else:
        uri = f"ws://{args.host}:{args.port}"
        ssl_context = None

    async with websockets.connect(uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context) as websocket:
        task = asyncio.create_task(record_microphone(websocket))
        task2 = asyncio.create_task(message(websocket))
        await asyncio.gather(task, task2)
        #print(f"Final Recognized Text: {recognized_text}")

        # 在完整识别后执行模型处理
        llm = Ollama(model="internlm2:7b-chat-v2.5-q4_K_S")
        all_output = llm.complete(recognized_text)
        print(all_output)

# 主函数，运行 WebSocket 客户端
if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(ws_client())
