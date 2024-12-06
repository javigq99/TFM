import numpy as np

def hz_to_mel(hz, htk=False):
    if htk:
        return 2595 * np.log10(1 + hz / 700)
    else:
        f_min = 0
        f_sp = 200 / 3
        mels = (hz - f_min) / f_sp
        min_log_hz = 1000
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27
        
        if np.isscalar(hz):
            if hz >= min_log_hz:
                return min_log_mel + np.log(hz / min_log_hz) / logstep
            else:
                return mels
        else:
            log_loc = hz >= min_log_hz
            mels[log_loc] = min_log_mel + np.log(hz[log_loc] / min_log_hz) / logstep
            return mels

def mel_to_hz(mel, htk=False):
    if htk:
        return 700 * (10**(mel / 2595) - 1)
    else:
        f_min = 0
        f_sp = 200 / 3
        min_log_hz = 1000
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27
        
        if np.isscalar(mel):
            if mel >= min_log_mel:
                return min_log_hz * np.exp(logstep * (mel - min_log_mel))
            else:
                return f_min + f_sp * mel
        else:
            log_loc = mel >= min_log_mel
            hz = np.zeros_like(mel, dtype=float)
            hz[~log_loc] = f_min + f_sp * mel[~log_loc]
            hz[log_loc] = min_log_hz * np.exp(logstep * (mel[log_loc] - min_log_mel))
            return hz

def get_mel_filterbanks(num_filters=128, nfft=2048, sample_rate=16000, 
                         fmin=0, fmax=None, htk=False, norm=1):
    if fmax is None:
        fmax = sample_rate / 2
    
    mel_low = hz_to_mel(fmin, htk)
    mel_high = hz_to_mel(fmax, htk)
    
    mel_points = np.linspace(mel_low, mel_high, num_filters + 2)
    hz_points = mel_to_hz(mel_points, htk)
    fft_bins = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)
    
    filterbanks = np.zeros((num_filters, int(nfft/2 + 1)))
    
    for m in range(1, num_filters + 1):
        left = fft_bins[m-1]
        center = fft_bins[m]
        right = fft_bins[m+1]
        
        for k in range(left, center):
            filterbanks[m-1, k] = (k - left) / (center - left)
        
        for k in range(center, right):
            filterbanks[m-1, k] = (right - k) / (right - center)
        
        if norm == 1:
            filterbanks[m-1] /= filterbanks[m-1].sum()
        elif norm == 2:
            filterbanks[m-1] /= filterbanks[m-1].max()
    
    return filterbanks

def compute_mfccs(signal, sample_rate=16000, n_mfcc=13, n_fft=2048, 
                  hop_length=None, win_length=None, window='hann', 
                  center=True, pad_mode='reflect', power=2.0, 
                  num_filters=128, fmin=0, fmax=None, htk=False, 
                  norm=1, dct_type=2):
    if hop_length is None:
        hop_length = n_fft // 4
    
    if win_length is None:
        win_length = n_fft
    
    if isinstance(window, str):
        if window == 'hann':
            window = np.hanning(win_length)
        else:
            raise ValueError(f"Unsupported window type: {window}")
    elif callable(window):
        window = window(win_length)
    
    if center:
        signal = np.pad(signal, int(n_fft // 2), mode=pad_mode)
    
    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=((signal.shape[0] - n_fft) // hop_length + 1, n_fft),
        strides=(signal.strides[0] * hop_length, signal.strides[0])
    )
    
    windowed_frames = frames * window
    spectrum = np.abs(np.fft.rfft(windowed_frames, n=n_fft))**power
    
    mel_filterbanks = get_mel_filterbanks(
        num_filters=num_filters, 
        nfft=n_fft, 
        sample_rate=sample_rate, 
        fmin=fmin, 
        fmax=fmax, 
        htk=htk, 
        norm=norm
    )
    mel_spectrum = np.dot(spectrum, mel_filterbanks.T)
    
    log_mel_spectrum = np.log(mel_spectrum + 1e-10)
    
    mfccs = np.zeros((log_mel_spectrum.shape[0], n_mfcc))
    
    # Implementación de la DCT (Transformada Discreta del Coseno)
    for i in range(log_mel_spectrum.shape[0]):
        mfccs[i] = np.cos(
            np.pi * np.arange(n_mfcc)[:, np.newaxis] * 
            (np.arange(num_filters) + 0.5) / num_filters
        ).dot(log_mel_spectrum[i])
    
    return mfccs.transpose(1,0)
