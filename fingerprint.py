import numpy as np
from matplotlib import pyplot as plt
import librosa
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion




#=====Parameters
DEFAULT_SR = 22050
CONST_WINSIZE = 4096
CONST_HOPSIZE = 2048
AMP_THRESHOLD = -15
NROWS_HASH = 32 
TIMEDELTA = 50 # in milliseconds


def find_peaks(spectrum, amp_threshold=AMP_THRESHOLD):
    """
    find peaks from spectrum
    """
    # an 4-connected neighborhood
    neighborhood = generate_binary_structure(2,1)

    # apply local maximum filter
    local_max = maximum_filter(spectrum, footprint=neighborhood)==spectrum

    # mask of background
    background = spectrum == 0
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # boolean mask
    detected_peaks = local_max ^ eroded_background

    # amplitudes and poistion
    amps = spectrum[detected_peaks]
    rows, columns = np.where(detected_peaks)

    # filter out the amps lower than threshold
    packed = zip(columns, rows, amps) # x, y, value
    filtered = [x for x in packed if x[2] >= amp_threshold]
    
    return filtered


def plot_peaks(spectrum, peaks):
    """
    plotting peaks
    """
    x = [v[0] for v in peaks]
    y = [v[1] for v in peaks]

    _, ax = plt.subplots()
    ax.imshow(spectrum)
    ax.scatter(x, y, s=1, c='r')
    plt.gca().invert_yaxis()
    plt.show()
    pass

def to_hash_list(peaks, nrows):
    result = [set() for _ in range(NROWS_HASH) ]
    
    frange = int((nrows/NROWS_HASH) + (0 if nrows%NROWS_HASH == 0 else 1))
    trange = (DEFAULT_SR * TIMEDELTA/1000)/CONST_HOPSIZE

    # inefficient way
    for t, f, _ in peaks:
        fidx = int(f/frange)
        tidx = int(t/trange)
        if tidx not in result[fidx]:
            result[fidx].add(tidx)

    # sorting elements
    for i in range(len(result)):
        result[i] = sorted(result[i])

    return result


def generate_audio_fingerprint(audiodata, samplerate, plot=False):
    # downsampling if the samplerate of audiodata is not same with DEFAULT_SR
    if samplerate != DEFAULT_SR:
        audiodata = librosa.resample(audiodata, samplerate, DEFAULT_SR)

    # normalize
    audiodata = librosa.util.normalize(audiodata)

    # generate Short-time Fourier Transform
    X = librosa.stft(audiodata, n_fft=CONST_WINSIZE, hop_length=CONST_HOPSIZE)
    C = librosa.amplitude_to_db(np.abs(X), ref=np.max)

    # use only a half of freq
    C = C[0:int(CONST_WINSIZE/4)]

    # find peaks on spectrum
    peaks = find_peaks(C, AMP_THRESHOLD)

    ## plotting for 
    if plot: plot_peaks(C, peaks)

    # generate hashlist
    hashlist = to_hash_list(peaks, len(C))

    return hashlist


#=======================================================MATCHING
def to_tuple(fingerprint):
    result = []
    for ih in range(len(fingerprint)):
        for n in fingerprint[ih]:
            result.append((n, ih))
    return result


def matching_hash(query, document):
    # for table size
    lenqry = max([len(l) for l in query if len(l) != 0])
    lendoc = max([l[-1] for l in document if len(l) != 0]) + 1

    # convert to tuple structure
    query = to_tuple(query)

    # initialise the table
    table = np.zeros((len(query), lendoc+(lenqry*2-2)))

    M = lenqry-1
    i = 0
    for n, h in query: 
        for v in (np.array(document[h]) - n):
            table[i,v+M] = 1
        i += 1

    return np.sum(table, axis=0)
