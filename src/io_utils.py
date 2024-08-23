import json
import numpy as np
import PIL.Image as PImage
import urllib.request as request
import wave

from math import exp
from PIL import ImageFilter as PImageFilter
from scipy.ndimage import convolve
from sklearn.cluster import KMeans


## Print Lists

def print_list_with_index(l):
  frmt = "{:>5},"*(len(l) - 1) + "{:>5}"
  ls = '[' + frmt.format(*range(10)) + ']\n'
  ls += '[' + frmt.format(*l) + ']\n'
  print(ls)


## Data Files

def object_from_json_url(url):
  with request.urlopen(url) as in_file:
    return json.load(in_file)


## Audio Files

def wav_to_list(wav_filename):
  with wave.open(wav_filename, mode="rb") as wav_in:
    if wav_in.getsampwidth() != 2:
      raise Exception("Input not 16-bit")

    nchannels = wav_in.getnchannels()
    nframes = wav_in.getnframes()
    nsamples = nchannels * nframes
    xb = wav_in.readframes(nframes)
    b_np = np.frombuffer(xb, dtype=np.int16)

    return [int(sum(b_np[b0 : b0 + nchannels]) / nchannels) for b0 in range(0, nsamples, nchannels)]

def list_to_wav(wav_array, wav_filename):
  xb = np.array(wav_array, dtype=np.int16).tobytes()
  with wave.open(wav_filename, "w") as wav_out:
    wav_out.setnchannels(1)
    wav_out.setsampwidth(2)
    wav_out.setframerate(44100)
    wav_out.writeframes(xb)


# Audio Analysis Functions

def logFilter(x, factor=3):
  if factor < 1:
    return x
  else:
    return np.exp(factor * np.log(x)) // np.power(10, factor*5)

def fft(samples, filter_factor=3):
  _fft = logFilter(np.abs(np.fft.fft(samples * np.hanning(len(samples))))[ :len(samples) // 2], filter_factor).tolist()
  num_samples = len(_fft)
  hps = (44100//2) / num_samples
  _freqs = [s * hps for s in range(num_samples)]
  return _fft, _freqs

def stft(samples, window_len=1024):
  _times = list(range(0, len(samples), window_len))

  sample_windows = []
  for s in _times:
    sample_windows.append(samples[s : s + window_len])

  sample_windows[-1] = (sample_windows[-1] + len(sample_windows[0]) * [0])[:len(sample_windows[0])]
  _ffts = [np.log(fft(s, filter_factor=0)[0]).tolist() for s in sample_windows]
  _, _freqs = fft(sample_windows[0], filter_factor=0)
  return _ffts, _freqs, _times

def cluster_fft_freqs(energy_freqs, freqs=None, *, top=50, clusters=6):
  if freqs is not None:
    energy_freqs = [(e, round(f)) for e,f in zip(energy_freqs, freqs)]

  fft_sorted = sorted(energy_freqs, key=lambda x: x[0], reverse=True)[:top]
  top_freqs = [[f[1]] for f in fft_sorted]

  kmeans = KMeans(n_clusters=clusters, n_init="auto").fit(top_freqs)
  return np.sort(kmeans.cluster_centers_, axis=0)[:, 0].astype(np.int16).tolist()

def ifft(fs):
  return np.fft.fftshift(np.fft.irfft(fs)).tolist()

def tone(freq, length_seconds, amp=4096, sr=44100):
  length_samples = length_seconds * sr
  t = range(0, length_samples)
  ham = np.hamming(length_samples)
  two_pi = 2.0 * np.pi
  return np.array([amp * np.sin(two_pi * freq * x / sr) for x in t] * ham).astype(np.int16).tolist()

def tone_slide(freq0, freq1, length_seconds, amp=4096, sr=44100):
  length_samples = length_seconds * sr
  t = range(0, length_samples)
  ham = np.hamming(length_samples)
  two_pi = 2.0 * np.pi
  f0 = min(freq0, freq1)
  f_diff = max(freq0, freq1) - f0
  return np.array([amp * np.sin(two_pi * (f0 + x / length_samples * f_diff) * x / sr) for x in t] * ham).astype(np.int16).tolist()


## Image Files

def update_pixels(mimg):
  def _update_pixels(pxs):
    if len(pxs) != mimg.size[0] * mimg.size[1]:
      raise Exception("array has wrong length")
    if not (type(pxs[0]) is int or type(pxs[0]) is tuple):
      raise Exception("array has wrong content type: must be int or tuple")
    if type(pxs[0]) is tuple and len(pxs[0]) != len(mimg.getbands()):
      raise Exception("array has wrong content format: number of channels must match original")
    if type(pxs[0]) is int and len(mimg.getbands()) != 1:
      pxs = [tuple([p]*len(mimg.getbands())) for p in pxs]
    mimg.putdata(pxs)
    mimg.pixels = list(mimg.getdata())
  return _update_pixels

def copy_image(mimg):
  super_copy = mimg.copy
  def _copy():
    cimg = super_copy()
    cimg.pixels = mimg.pixels[:]
    cimg.update_pixels = update_pixels(cimg)
    cimg.copy = copy_image(cimg)
    return cimg
  return _copy

def open_image(path):
  mimg = PImage.open(path)
  mimg.pixels = list(mimg.getdata())

  mimg.copy = copy_image(mimg)
  mimg.update_pixels = update_pixels(mimg)
  return mimg


## Image Analyssis

def constrain_uint8(v):
  return int(min(max(v, 0), 255))

def blur(img, rad=1.0):
  cimg = img.copy()
  bimg = img.filter(PImageFilter.GaussianBlur(rad))
  cimg.update_pixels(list(bimg.getdata()))
  return cimg

def edges_rgb(img, rad=1.0):
  bimg = blur(img, rad)
  pxs = img.pixels
  bpxs = bimg.pixels

  bdiffpx = []
  for (r0,g0,b0), (r1,g1,b1) in zip(bpxs, pxs):
    bdiffpx.append((
      constrain_uint8(exp(r1-r0)),
      constrain_uint8(exp(g1-g0)),
      constrain_uint8(exp(b1-b0)),
    ))
  bimg.update_pixels(bdiffpx)
  return bimg

def edges(img, rad=1.0):
  bimg = blur(img, rad)
  pxs = img.pixels
  bpxs = bimg.pixels

  bdiffpx = []
  for (r0,g0,b0), (r1,g1,b1) in zip(bpxs, pxs):
    bdiffpx.append(constrain_uint8(exp(r1-r0)))

  bimg.update_pixels(bdiffpx)
  return bimg

def conv2d(img, kernel):
  pxs = np.array(img.convert("L").getdata()).reshape(img.size[1], -1).astype(np.uint8)
  krl = np.array(kernel)
  cpxs = convolve(pxs, krl).reshape(-1).astype(np.uint8).tolist()

  # returns plain PImage
  w,h = img.size
  nimg = PImage.new(img.getbands(), (w, h))
  nimg.putdata(cpxs)

  return nimg

def conv2drgb(img, kernel):
  pxs = np.array(img.getdata()).reshape(img.size[1], -1, 3).astype(np.uint8)
  krl = np.repeat(np.array(kernel).reshape(len(kernel), len(kernel[0]), 1), 3, axis=2)
  _cpxs = convolve(pxs, krl).reshape(-1, 3).astype(np.uint8).tolist()
  cpxs = [(r,g,b) for r,g,b in _cpxs]

  # returns plain PImage
  w,h = img.size
  nimg = PImage.new(img.getbands(), (w, h))
  nimg.putdata(cpxs)

  return nimg
