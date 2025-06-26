import pyaudio

pa = pyaudio.PyAudio()

print("可用录音设备列表：\n")
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:  # 只显示输入设备
        print(f"设备索引: {i}")
        print(f"  名称: {info['name']}")
        print(f"  输入通道数: {info['maxInputChannels']}")
        print(f"  默认采样率: {info['defaultSampleRate']}")
        print("-" * 30)

pa.terminate()