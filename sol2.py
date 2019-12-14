import numpy as np
from scipy.io import wavfile as wav
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from imageio import imread
from skimage.color import rgb2gray


RGB_SHAPE = 3
GRAY_REP = 1
NORM_FACTOR = 255
DER = [[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]]


def read_image(filename, representation):
    """
    Reads an image file and converts it into a given representation.
    :param filename: The image filename.
    :param representation: Representation code - (1) gray scale image (2) RGB image.
    :return: The converted image normalized to the range [0, 1].
    """
    im = imread(filename)
    if len(im.shape) == RGB_SHAPE and representation == GRAY_REP:
        im = rgb2gray(im)
        return im.astype(np.float64)
    im_float = im.astype(np.float64)
    im_float /= NORM_FACTOR
    return im_float


def DFT_matrix(n, sign):
    """
    :param n: The signal length.
    :param sign: The sign of the power of the exponent, will be -1 for DFT and +1 for IDFT.
    :return: the DFT matrix.
    """
    omega = np.exp((sign * 2 * np.pi * 1j) / n)
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    return omega ** (i * j)


def DFT(signal):
    """
    :param signal: A signal to calculates its DFT, given as an array.
    :return: The DFT of the given signal.
    """
    dft_matrix = DFT_matrix(signal.shape[0], -1)
    return dft_matrix.dot(signal).astype(np.complex128)


def IDFT(fourier_signal):
    """
    :param fourier_signal: A fourier signal to calculates its IDFT, given as an array.
    :return: The IDFT of the given fourier signal.
    """
    n = fourier_signal.shape[0]
    dft_matrix = DFT_matrix(n, 1)
    res = dft_matrix.dot(fourier_signal).astype(np.complex128) / n
    return np.real_if_close(res)


def DFT2(image):
    """
    :param image: An image to calculates its 2D DFT, given by a matrix (2D array).
    :return: The DFT of the given image.
    """
    res = DFT(image)
    trans_res = np.transpose(res)
    trans_res = DFT(trans_res)
    return np.transpose(trans_res)


def IDFT2(fourier_image):
    """
    :param image: A fourier image to calculates its 2D IDFT, given by a matrix (2D array).
    :return: The IDFT of the given fourier image.
    """
    res = IDFT(fourier_image)
    trans_res = np.transpose(res)
    trans_res = IDFT(trans_res)
    return np.transpose(trans_res)


def change_rate(filename, ratio):
    """
    Creates a new wav file with the given audio after the rate change.
    :param filename: The wav filename to change its rate.
    :param ratio: The duration change.
    """
    rate, data = wav.read(filename)
    wav.write('change_rate.wav', int(rate * ratio), data)


def resize(data, ratio):
    """
    Resize the original sampled points array according to the given ratio, by using DFT.
    :param data: The original sampled point, given by as an array.
    :param ratio: The duration change.
    :return: The new sampled points array.
    """
    dft = DFT(data)
    shifted_dft = np.fft.fftshift(dft)
    middle = np.where(shifted_dft == dft[0])[0][0]
    size = data.shape[0]
    new_size = int(size / ratio)
    if ratio >= 1:
        start = int(middle - new_size / 2)
        new_dft = shifted_dft[start: start + new_size]
    else:
        zeros = new_size - size
        new_dft = np.pad(shifted_dft, (int(zeros / 2), int(zeros / 2) + zeros % 2),
                         'constant', constant_values=0)
    new_dft = np.fft.ifftshift(new_dft)
    return IDFT(new_dft).astype(data.dtype)


def change_samples(filename, ratio):
    """
    Creates a new wav file with the given audio after the resizing process.
    :param filename: The wav filename to change its rate.
    :param ratio: The duration change.
    :return: The new samples points array.
    """
    rate, data = wav.read(filename)
    new_data = resize(data, ratio).astype(np.float64)
    wav.write('change_samples.wav', rate, new_data / np.max(new_data))
    return new_data


def resize_spectrogram(data, ratio):
    """
    Resize the given audio data by using its spectrogram.
    :param data: The original sampled point, given by as an array.
    :param ratio: The duration change.
    :return: The new samples points array.
    """
    spec = stft(data)
    new_spec = []
    for i in range(spec.shape[0]):
        new_spec.append(resize(spec[i], ratio))
    return istft(np.array(new_spec))


def resize_vocoder(data, ratio):
    """
    Resize the given audio data by using its vocoded spectrogram.
    :param data: The original sampled point, given by as an array.
    :param ratio: The duration change.
    :return: The new samples points array.
    """
    spec = stft(data)
    vocoded_spec = phase_vocoder(spec, ratio)
    return istft(vocoded_spec)


def conv_der(im):
    """
    Calculates the given image derivatives magnitude by using convolution.
    :param im: Gray-scale image to calculate its derivatives magnitude.
    :return: The magnitude of the given image derivatives, as a matrix.
    """
    x_der = signal.convolve2d(im, DER, 'same')
    y_der = signal.convolve2d(im, np.transpose(DER), 'same')
    return np.sqrt(x_der ** 2 + y_der ** 2)


def fourier_der(im):
    """
    Calculates the given image derivatives magnitude by using fourier transform.
    :param im: Gray-scale image to calculate its derivatives magnitude.
    :return: The magnitude of the given image derivatives, as a matrix.
    """
    n, m = im.shape
    half_n, half_m = int(n / 2), int(m / 2)
    dft = DFT2(im)
    shifted_dft = np.fft.fftshift(dft)
    cols, rows = np.meshgrid(np.arange(-half_n, half_n), np.arange(-half_m, half_m))
    shifted_x_der_dft = (2j * np.pi / n) * np.transpose(cols) * shifted_dft
    shifted_y_der_dft = (2j * np.pi / m) * np.transpose(rows) * shifted_dft
    x_der_dft, y_der_dft = np.fft.ifftshift(shifted_x_der_dft), np.fft.ifftshift(shifted_y_der_dft)
    x_der, y_der = IDFT2(x_der_dft), IDFT2(y_der_dft)
    return np.sqrt(np.abs(x_der) ** 2 + np.abs(y_der) ** 2)


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    time_steps = np.arange(spec.shape[1]) * ratio
    time_steps = time_steps[time_steps < spec.shape[1]]

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec
