from audioidentification import *

def main():
    fingerprintBuilder('./database/', './fingerprints/')
    audioIdentification('./queryset', './fingerprints/', './output.txt')

if __name__ == '__main__':
    main()


