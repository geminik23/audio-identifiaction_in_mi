# Audio Identification System. 

## Task
The task is to implement and test an audio identifiaction system.
you may take a published paper and attempt to reimplement it, such as one of the approaches mentioned in the lectures.

To aid in automatic testing, your code must be callable in two parts as follows:
fingerprintBuilder(/path/to/database/,/path/to/fingerprints/)
audioIdentification(/path/to/queryset/,/path/to/fingerprints/,/path/to/output.txt)

The format of the output.txt file will include one line for each query audio recording, using the
following format:
query audio 1.wav database audio x1.wav database audio x2.wav database audio x3.wav
query audio 2.wav database audio y1.wav database audio y2.wav database audio y3.wav
...

## Dataset
To aid in developing and testing your code, you can download a subset of 300 classical, jazz, and pop
audio clips from the GTZAN dataset.

## Requirements
```
librosa
numpy
matplotlib
scipy
```

## Run
```sh
python3 main.py
```



