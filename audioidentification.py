import os
import numpy as np
import librosa
from fingerprint import generate_audio_fingerprint, matching_hash
import sqlite3
import json


class AudioIdentificationDB:
    def __init__(self, path):
        self._path = path
        filepath = os.path.join(path, "afpdoc.db")
        self._conn = sqlite3.connect(filepath)
        self.check()

    def check(self):
        self._conn.execute(''' CREATE TABLE IF NOT EXISTS documents( 
                filename TEXT NOT NULL, 
                hashlist TEXT NOT NULL,
                UNIQUE(filename)
                )''') 

    def restore_fingerprint_info(self, fileinfo):
        fname = fileinfo.name
        hashlist = json.dumps(fileinfo.fingerprint)
        # insert or replace if exists
        self._conn.execute('''insert or replace into documents values (?,?)''', (fname, hashlist))
        self._conn.commit()

        pass

    def load_all_fingerprints(self):
        c = self._conn.cursor()
        c.execute('select * from documents')
        records = c.fetchall()
        c.close()

        return [(f, json.loads(j)) for f, j in records]

    def close(self):
        self._conn.close()
        pass








class WavInfo:
    pass

def list_all_wavfiles(path): 
    """
    load all wav files in directory(@path)
    """
    result = []
    for entry in os.scandir(path):
        if not entry.name[-4:] == '.wav': continue
        wavinfo = WavInfo()
        wavinfo.path = entry.path
        wavinfo.name = entry.name
        wavinfo.sr = None

        result.append(wavinfo)
    return result
        








def fingerprintBuilder(db_path, fingerprint_path):
    # database
    db = AudioIdentificationDB(fingerprint_path)

    # load all wav files
    wavfiles = list_all_wavfiles(db_path)
    # wavfiles = [wavfiles[0]]

    for fileinfo in wavfiles:
        # load audio file
        X, sr = librosa.load(fileinfo.path)

        print('generating documents fingerprint of {}...'.format(fileinfo.name))
        # generate audio fingerprint
        fileinfo.fingerprint = generate_audio_fingerprint(X, sr, False)

        # save to database
        db.restore_fingerprint_info(fileinfo)
        pass
    db.close()


def audioIdentification(query_path, fingerprint_path, out_filename):
    # database
    db = AudioIdentificationDB(fingerprint_path)

    # load all fingerprints
    documents = db.load_all_fingerprints()

    # query files
    queryfiles = list_all_wavfiles(query_path)
    with open(out_filename, 'w') as f:
        for fileinfo in queryfiles:
            print('matching {}...'.format(fileinfo.name))
            X, sr = librosa.load(fileinfo.path)

            # denoise using simple moving average
            win_len = 11
            temp = np.r_[X[win_len-1:0:-1], X, X[-2:-win_len-1:-1]] # padding each side
            w = np.ones(win_len,'d')
            X = np.convolve(w/w.sum(), temp)
            

            # generate audio fingerprint
            fileinfo.fingerprint = generate_audio_fingerprint(X, sr, False)

            # matching test
            r = []
            for doc in documents:
                matchs = matching_hash(fileinfo.fingerprint, doc[1])
                r.append(max(matchs))
            sorted_idx = np.argsort(np.array(r))
            out = '{} {} {} {}\n'.format(fileinfo.name, 
                    documents[sorted_idx[-1]][0],
                    documents[sorted_idx[-2]][0],
                    documents[sorted_idx[-3]][0])
            f.write(out)
            

    db.close()
    pass

