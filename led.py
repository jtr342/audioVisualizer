from __future__ import print_function
from __future__ import division

import platform
import numpy as np
import config

from rpi_ws281x import *
strip = Adafruit_NeoPixel(config.N_PIXELS, config.LED_PIN,
                                    config.LED_FREQ_HZ, config.LED_DMA,
                                    config.LED_INVERT, config.BRIGHTNESS)
strip.begin()


_gamma = np.load(config.GAMMA_TABLE_PATH)
"""Gamma lookup table used for nonlinear brightness correction"""

_prev_pixels = np.tile(253, (3, config.N_PIXELS))
"""Pixel values that were most recently displayed on the LED strip"""

pixels = np.tile(1, (3, config.N_PIXELS))
"""Pixel values for the LED strip"""

_is_python_2 = int(platform.python_version_tuple()[0]) == 2

def _update_pi():
    global pixels, _prev_pixels
    pixels = np.clip(pixels, 0, 255).astype(int)
    p = _gamma[pixels] if config.SOFTWARE_GAMMA_CORRECTION else np.copy(pixels)
    r = np.left_shift(p[0][:].astype(int), 8)
    g = np.left_shift(p[1][:].astype(int), 16)
    b = p[2][:].astype(int)
    rgb = np.bitwise_or(np.bitwise_or(r, g), b)
    for i in range(config.N_PIXELS):
        if np.array_equal(p[:, i], _prev_pixels[:, i]):
            continue
            
        strip._led_data[i] = int(rgb[i])
    _prev_pixels = np.copy(p)
    strip.show()



def update():
    _update_pi()
   
if __name__ == '__main__':
    import time
 
    pixels *= 0
    pixels[0, 0] = 255 
    pixels[1, 1] = 255  
    pixels[2, 2] = 255  
    print('Starting LED strand test')
    while True:
        pixels = np.roll(pixels, 1, axis=1)
        update()
        time.sleep(.1)
