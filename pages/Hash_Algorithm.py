import binascii
import streamlit as st
import streamlit_nested_layout
from imutils import paths
from tkinter import Tk, filedialog
import scipy.fftpack
import numpy as np
import argparse
import time
import sys
from PIL import Image
import os
import base64

# Setting page to utilize Full Body Size
st.set_page_config(page_title = "Hash Algorithn", layout="wide")

st.markdown("# <div style=\"text-align: center;\">Implementing a Hash Algorithm</div>", unsafe_allow_html=True)
" "
" "

__, cont, __ = st.columns([1,5,1])
 
cont.markdown("""
#### What is image hashing?

Image hashing or perceptual hashing is the process of:
- Examining the contents of an image
- Constructing a hash value that uniquely identifies an input image based on the contents of an image
By utilizing image hashing algorithms we can find near-identical images in constant time, or at worst, O(lg n) time when utilizing the proper data structures.

---
""")

cont.markdown("""
#### Step #1: Convert to grayscale

The first step in our image hashing algorithm is to convert the input image to grayscale and discard any color information.
Discarding color enables us to:
- Hash the image faster since we only have to examine one channel
- Match images that are identical but have slightly altered color spaces (since color information has been removed)
""")
cont.markdown("##### Illustration:")
img = cont.file_uploader("Example image:", type= ["jpeg", "png", "jpg"])
x, y, z = cont.columns([3,1,3])
greyScaleImg = None
resized = None
diff = None
if img is not None:
    x.image(img)
    x.markdown("***Original Image***")
    img = Image.open(img)
    greyScaleImg = img.convert('L')
    z.image(greyScaleImg)
    z.write("***Greyscale Image***")

cont.markdown("""
#### Step #2: Resize

Now that our input image has been converted to grayscale, we need to squash it down to 9×8 pixels, ignoring the aspect ratio. For most images + datasets, the resizing/interpolation step is the slowest part of the algorithm.
We squash the image down to 9×8 and ignore aspect ratio to ensure that the resulting image hash will match similar photos regardless of their initial spatial dimensions.
""")
cont.markdown("##### Illustration:")
size_input, __, resized_output = cont.columns([3,1,3])
hash_len = size_input.number_input("Hash Length", min_value=1, value=8)
high_freq_factor = size_input.number_input("DCT Truncation Factor", min_value=1, value=4)
hashSize = hash_len * high_freq_factor
if greyScaleImg is not None:
    resized = greyScaleImg.resize((int(hashSize), int(hashSize)), Image.LANCZOS)
    resized_output.image(resized, caption = "Resized Image ({}×{})".format(hashSize, hashSize))
    resized_output.write("<style>div.e115fcil1 img {image-rendering: pixelated;}</style>", unsafe_allow_html=True)
    
    
cont.markdown("""
#### Step #3: Compute the difference
- ***Reduce the DCT***: While the DCT is 32x32, just keep the top-left 8x8. Those represent the lowest frequencies in the picture.
- ***Compute the average value***: Like the Average Hash, compute the mean DCT value (using only the 8x8 DCT low-frequency values and excluding the first term since the DC coefficient can be significantly different from the other values and will throw off the average).
- ***Further reduce the DCT***. This is the magic step. Set the 64 hash bits to 0 or 1 depending on whether each of the 64 DCT values is above or below the average value. The result doesn't tell us the actual low frequencies; it just tells us the very-rough relative scale of the frequencies to the mean. The result will not vary as long as the overall structure of the image remains the same; this can survive gamma and color histogram adjustments without a problem.
""")
if resized is not None:
    x, y, z = cont.columns([4,1,5])
    pixels = np.asarray(resized)
    x.write("Pixels:")
    x.write(pixels)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
    dctlowfreq = dct[:int(hash_len), :int(hash_len)]
    med = np.median(dctlowfreq)
    y.write("med:")
    y.write(med)
    diff = dctlowfreq > med
    z.write("diff:")
    z.write(diff)

cont.markdown("""
#### Step #4: Build the hash
Set the 64 bits into a 64-bit integer. The order does not matter, just as long as you are consistent. To see what this fingerprint looks like, simply set the values (this uses +255 and -255 based on whether the bits are 1 or 0) and convert from the 32x32 DCT (with zeros for the high frequencies) back into the 32x32 image.
""")
if diff is not None:
    f = diff.flatten()
    l = f.shape[0] - 1
    hashValue = sum([2**(l - i) for i, v in enumerate(f) if v])
    cont.write(f"**Final hash: `{hashValue:x}`**")
