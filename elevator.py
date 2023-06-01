import pyaudio
import wave
import cv2
import auditok
import os

def openWindow():
    """
    Load the elevator buttons image.

    ### input
        This function has no input

    ### output
        - img : np 3d array

            image of elevator buttons
    """

    img = cv2.imread("elevator.jpg")
    cv2.imshow("elevator", img)
    return img

def voiceDetect(filename, savePath = "./audio/", 
                chunk = 1024, sample_format = pyaudio.paInt16, 
                channels = 2, fs = 44100, seconds = 5):
    """
    Press any key to record the voice with a microphone and generate a wav file.

    ### input
        - img : np 3d array

            image of elevator buttons

        - filename : str

            name of output sound file

    ### output
        - filePath : str

            output wav file path
    """

    p = pyaudio.PyAudio()
    cv2.waitKey(0)
    frames = []
    stream = p.open(format=sample_format, channels=channels,
                     rate=fs, frames_per_buffer=chunk, input=True)
    print("start recording")


    for _ in range(int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
        
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("end recording")

    wf = wave.open(savePath + filename + ".wav", 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    region = auditok.load(savePath + filename + ".wav")
    splited = region.split(min_dur = 0.3, max_dur = 1.5, max_silence = 0.1, energy_threshold = 55)
    commands = []
    for i, r in enumerate(splited):
        commands.append(r.save(savePath + "command_" + str(i) + ".wav"))

    return os.listdir(savePath)

def closeWindow():
    """
    Close the image.
    """

    cv2.destroyAllWindows()

def floorChoose(img, floor):
    """
    Press the floor button.

    ### input
        - img : np 3d array

            image of elevator buttons

        - floor : int

            target floor button to press

    ### output
        This function has no output
    """

    center = (210 - (floor % 2) * 120, 525 - int((floor - 1) / 2) * 105)
    cv2.circle(img, center, 40, (193, 244, 255), -1)
    cv2.putText(img, str(floor + int("0")), (center[0] - 15, center[1] + 17),
                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, (25, 25, 25), 5)
    cv2.imshow("elevator", img)
    return


# test
img = openWindow()
filepath = voiceDetect("test")
print(filepath)
floorChoose(img, 4)
cv2.waitKey(0)
closeWindow()