# -*- coding: utf-8 -*-
import numpy as np
import pyaudio
from pydub import AudioSegment, effects
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
'''
#快速傅里叶变换 (FFT), 即利用计算机计算离散傅里叶变换（DFT)的高效、快速计算方法的统称，简称FFT。
#采用这种算法能使计算机计算离散傅里叶变换所需要的乘法次数大为减少，特别是被变换的抽样点数N越多，FFT算法计算量的节省就越显著。
'''

'''
#离散傅里叶变换（DFT），是傅里叶变换在时域和频域上都呈现离散的形式，将时域信号的采样变换为在离散时间傅里叶变换（DTFT）频域的采样。
在形式上，变换两端（时域和频域上）的序列是有限长的，而实际上这两组序列都应当被认为是离散周期信号的主值序列。
即使对有限长的离散信号作DFT，也应当将其看作经过周期延拓成为周期信号再作变换。
在实际应用中通常采用快速傅里叶变换以高效计算DFT。
'''
# ------------------------------------两条线模式
# 音频部分
p = pyaudio.PyAudio()  # 实例化PyAudio ，它设置portaudio系统
music = input("请输入歌曲文件名：")
sound = AudioSegment.from_file(file=music + ".wav")  # 加载声音文件
left = sound.split_to_mono()[0]  # 左声道
fs = left.frame_rate  # 设置帧率为默认(11.025 kHz)
size = len(left.get_array_of_samples())  # 以（数值）样本数组的形式记录原始音频数据。并使用len返回字符数
channels = left.channels  # 监听频率
stream = p.open(
    format=p.get_format_from_width(left.sample_width, ),  # 返回指定宽度的PortAudio格式常数
    channels=channels,
    rate=fs,
    output=True,  # 使用默认输出设备
)
stream.start_stream()  # 开始处理音频流，该音频流将反复调用回调函数，直到该函数返回为止

# 窗口部分
fig = plt.figure()  # 实例化figure，创建默认图像
ax1 = fig.subplots()  # 实例化subplots，创建子图
ax1.set_ylim(0, 3)  # 设置子图y轴视图限制（底部和顶部）
ax1.set_axis_off()  # 使用set_axis_off()方法关闭子图中的轴
window = int(0.02 * fs)  # 刷新频率 20ms

g_windows = window // 8  # 调整曲线光滑程度

# 窗口设置部分
f = np.linspace(20, 20 * 1000, g_windows)  # 在音频数据流指定的间隔内返回均匀间隔的数字
lf1, = ax1.plot(f, np.zeros(g_windows), lw=1)  # 根据返回的数据流绘出相应图像
lf1.set_antialiased(True)  # 绘图中抗锯齿
color_grade = ['blue', 'gold', 'red', 'black']  # 对不同范围的图像（波段）设置颜色


# 绘制动态图像
def update(frames):  # 动态生成波形曲线
    if stream.is_active():  # stream	用于同时输入和输出的PortAudio流（使用NumPy）
        # 媒体流，用于监听当前音频
        slice = left.get_sample_slice(frames, frames + window)  # 使用音频文件在类slice（）中创建对象并传入波形曲线（frames）和刷新频率
        data = slice.raw_data  # 记录图像
        stream.write(data)  # 绘制图像
        y = np.array(slice.get_array_of_samples()) / 20000  # 归一化
        yft = np.abs(np.fft.fft(y)) / g_windows  # 将傅里叶变换结果返回至窗口
        grade = int(max(yft[:g_windows]) - min(yft[:g_windows]))  # 设置窗口图像显示最大值与最小值
        if 0 <= grade < len(color_grade):  # 设置不同范围图像颜色
            lf1.set_color(color_grade[grade])
        lf1.set_ydata(yft[:g_windows])
    return lf1,  # 返回最终图像


ani = FuncAnimation(fig, update, frames=range(0, size, window), interval=0, blit=True)  # 动态模拟
plt.show()
