import speech_recognition as sr
from gtts import gTTS
import playsound

def speak(text):
    tts = gTTS(text=text, lang='ko')
    #파일 이름 설정
    filename='mmyeong.mp3'
    tts.save(filename)
    playsound.playsound(filename)

speak('TTS 테스트 중입니다 확인 결과 잘 나오는 것을 알 수 있어요')