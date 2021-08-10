'''
Simple code to prevent your computer from timing out
Picked key that does not mess with Mario Kart whatsoever
Was leaning for F9 but that messed with it so I used shift
May be inconvenient if you wish to do something else on your computer
Between races
'''
from pynput.keyboard import Key, Controller  # Needed libraries
import time
keyboard = Controller()
while True:  # Prevent Time-Out loop: Pressing shift every 5 sec
    time.sleep(5)
    keyboard.press(Key.shift)
    time.sleep(0.125)
    keyboard.release(Key.shift)
