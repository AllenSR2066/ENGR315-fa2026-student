"""
CULMINATING ASSIGNMENT 3: Musical Note Detection

Create a program which detects the note being played from the given audio file based on its frequency.
It is up to you to employ an appropriate methodology to process the audio signal and detect the note.
The relevant data file is named "Guitar-note-1.wav" and is located in data > musical-notes.
In the same directory, we also provide a reference for the corresponding frequencies of each note.

Your code needs to be successful as well as sufficiently commented/documented to receive full credit.
We have provided a possible method for converting the .wav file to usable data below.
All file paths need to be relative and execute from this file's directory;
the Visual Studio Code setting "Python > Terminal: Execute in File Dir" 
must be enabled in File > Preferences > Settings.

See Canvas for additional requirements.
"""

from scipy.io import wavfile

# Read WAV file (replace "None")
None, None = wavfile.read(None)
