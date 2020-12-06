Justin Rabe, 820663292 Raspberry Pi Audio Visualizer
The source code and prototype were built following along to Scott Lawson's tutorial on how to build an Audo Reactive LED strip, listed here: 

https://github.com/scottlawsonbc/audio-reactive-led-strip

Scott Lawson's tutorial also includes several other modes and hardware, and thus I only included the raspberry pi portions.
The Audio Visualizer works by taking in input data from a USB microphone and converting that data
into usable data for our LEDs. It utilizes these libraries: numpy (for tables), scipy (for calculations), and pyaudio for 
the audio stuff. 

Overall, the flow for the our project goes from:

Speaker -> Microphone Input -> conversions and calculations -> LED output

My original plan with this project was to utilize the Spotify API as input, which is something I will be doing in the future. For now, with
the time constraints as well as pressure overall from other projects, I decided to go with a simpler design that simply takes in microphone input. 



