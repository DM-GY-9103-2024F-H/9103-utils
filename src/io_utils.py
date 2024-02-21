import json
import numpy as np
import PIL.Image as Image
import urllib.request as request
import wave


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
    if wav_in.getnchannels() != 1:
      raise Exception("Input mono")

    xb = wav_in.readframes(wav_in.getnframes())
    return list(np.frombuffer(xb, dtype=np.int16))

def list_to_wav(wav_array, wav_filename):
  xb = np.array(wav_array, dtype=np.int16).tobytes()
  with wave.open(wav_filename, "w") as wav_out:
    wav_out.setnchannels(1)
    wav_out.setsampwidth(2)
    wav_out.setframerate(44100)
    wav_out.writeframes(xb)


## Image Files

def get_pixels(input):
  if type(input) is str:
    mimg = Image.open(input)
  elif isinstance(input, Image.Image):
    mimg = input
  else:
    raise Exception("wrong input type")
  return  list(mimg.getdata())

def get_Image(input, width=None, height=None):
  if type(input) is str:
    mimg = Image.open(input)
  elif type(input) is list:
    pxs = input
    num_channel = 1
    if type(pxs[0]) is list or type(pxs[0]) is tuple:
      num_channel = len(pxs[0])

    img_mode = "L"
    if num_channel == 3:
      img_mode = "RGB"
    elif num_channel == 4:
      img_mode = "RGBA"

    w,h = width, height
    if width == None:
      w = int(len(pxs) ** 0.5)
      h = w
    elif height == None:
      h = int(len(pxs) / w)

    pxs = pxs[:w * h]

    mimg = Image.new(img_mode, (w, h))
    mimg.putdata(pxs)
  else:
    raise Exception("wrong input type")
  return mimg
