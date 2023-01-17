Pytorch에서는 오디오 데이터를 뻥튀기하는 다양한 방법들을 제공한다.
여기서 "뻥튀기한다"라는 것은, 한정적인 오디오 데이터를 이런저런 방법으로 조작해 새로운 오디오 데이터처럼 조작한다고 생각하면 될 것 같다.  


``` python
import math

from IPython.display import Audio
import matplotlib.pyplot as plt

from torchaudio.utils import download_asset

SAMPLE_WAV = download_asset("tutorial-assets/steam-train-whistle-daniel_simon.wav")
SAMPLE_RIR = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")
SAMPLE_SPEECH = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav")
SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
```

```python
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)
    
    
def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, _ = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)
```

### 1) Applying SoX effects

	torchaudio.sox_effects.apply_effects_tensor()
	torchaudio.sox_effects.apply_effects_file()
    
Linux에서는 SoX(Sound eXchange)를 사용해 오디오에 다양한 효과를 적용할 수 있는데, torchaudio에서도 이 기능을 동일하게 제공한다. Tensor 형태로 표현한 오디오와, file object로 표현한 오디오 모두에 적용할 수 있다.

SoX에서 제공하는 효과(effects)들은 공식 문서에서 확인할 수 있다.

https://sox.sourceforge.net/sox.html#DIAGNOSTICS

```python
waveform1, sample_rate1 = torchaudio.load(SAMPLE_WAV)

# list[list[str]]의 형태

effects = [
    ["lowpass", "-1", "300"],
    ["speed", "0.8"],
    ["rate", f"{sample_rate1}"],
    ["reverb", "-w"]
]

## lowpass filter : 차단 주파수(300) 이하의 주파수 신호만 통과시키는 필터. (-1:single-pole, -2:double-pole)
## reverb : 반향(울림) 효과
```

이번 예제에서는 1) 주파수를 제한하고, 2) 속도를 늦췄으며, 3) 반향 효과를 주어 오디오를 변형시켰다.
원본과 어떤 차이가 있는지 확인해보자.

```python
Audio(waveform1, rate=sample_rate1)
```

<audio controls >
 <source src="https://docs.google.com/uc?export=open&id=1QC39H__aXe-enba0BRPMnbi_osk74kdy" type='audio/mp3' />
</audio>

https://drive.google.com/file/d/1QC39H__aXe-enba0BRPMnbi_osk74kdy/view?usp=share_link

<audio controls >
 <source src="https://docs.google.com/uc?export=open&id=1WPHpp5ogr9RObpafP56ui2xTlybJnoYp" type='audio/mp3' />
</audio>

https://drive.google.com/file/d/1WPHpp5ogr9RObpafP56ui2xTlybJnoYp/view?usp=share_link

### 2) Convolution reverb

Convolution reverb는 깨끗한 오디오를 다른 환경에서 녹음된 것처럼 변형시켜주는 기능이다. 이번 예제에서는 RIR (Room Impulse Response)을 사용해 오디오가 회의실에서 녹음된 것과 같은 효과를 줄 것이다.

다음은 회의실과 같은 환경에서 박수를 친 오디오이다.
```python
rir_raw, sample_rate = torchaudio.load(SAMPLE_RIR)
Audio(rir_raw, rate=sample_rate)
```
<audio controls >
 <source src="https://docs.google.com/uc?export=open&id=1FraqQJ4M9bUfmbHe2nFJtD1EVZfDl15d" type='audio/mp3' />
</audio>

https://drive.google.com/file/d/1FraqQJ4M9bUfmbHe2nFJtD1EVZfDl15d/view?usp=share_link

이 원본으로부터 1) 박수소리(main impulse) 부분만 잘라내고, 2) 이를 normalize한 뒤, 3) 시간축을 기준으로 뒤집는다(flip).

```python
# 소리가 나는 시간(박수) ~ 진동이 끝나는 시간까지 슬라이싱 하는 것이 RIR의 핵심!
rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)] # 박수 부분만
rir_normalized = rir / torch.norm(rir, p=2) # torch.norm : returns vector norm of tensor.

## 벡터 노름이라고하면, 어떤 벡터를 길이나 사이즈같은 양적인 수치로 mapping하기 위한 함수이다.
## 벡터(텐서)를 그것의 vector norm으로 나눠주면 unit vector, 즉 방향은 그대로고 크기는 1인 벡터가 된다.
## 방향만이 중요할 때 크기를 통일시키는 작업.

RIR = torch.flip(rir_normalized, [1])

plot_waveform(rir_normalized, sample_rate, title="Room Impulse Response")
plot_waveform(RIR, sample_rate, title="rir flipped along time axis")
```


