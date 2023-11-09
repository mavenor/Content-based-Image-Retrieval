import streamlit as st
import streamlit_nested_layout
import os
import pandas as pd
import imagehash
import hashes
import time
import glob
import os
from PIL import Image

# Setting page to utilize Full Body Size
st.set_page_config(layout="wide")
st.markdown("# <div style=\"text-align: center;\">Hashing a Directory of Images</div>", unsafe_allow_html=True)
" "
" "

__, cont, __ = st.columns([1,5,1])

path = cont.text_input("Enter path of directory containing images:")
btn = cont.button("Hash Folder")
method = cont.radio("Choose hashing method:", ('Parallel', 'Serial'))

if 'hashStatus' not in st.session_state:
    st.session_state.hashStatus: bool = False
    
progMsg = cont.markdown("")


if path and btn:
    if method == "Serial":
        x = hashes.hashes(path)
        start = time.time()
        st.session_state.hashStatus, st.session_state.hashValues = x.compute_hashes_serial()
        end = time.time()
        progMsg.text("Time taken = " + str(end - start))

    elif method == "Parallel":

        x = hashes.hashes(path)
        start = time.time()
        st.session_state.hashStatus, st.session_state.hashValues = x.compute_hashes_parallel(num_workers = os.cpu_count())
        end = time.time()
        progMsg.text("Time taken = " + str(end - start))


cont.write(" ")

if st.session_state.hashStatus:
    cont.write("✅ Hashing Complete! 😎")
    cont.write(" ")

    dfFrame, __, exportBtns = cont.columns([10,1,6])
    #Panda Data Frame
    df = pd.DataFrame(st.session_state.hashValues, columns=["Image", "Hash Value", "Path"]) 
    dfFrame.write("Hash Table:")
    dfFrame.dataframe(df)

    # CSV
    csvBtn = exportBtns.button("Export to csv")
    if csvBtn:
        df.to_csv(r"./output/test.csv")

    # Pickle
    pklBtn = exportBtns.button("Pickle the object")
    if pklBtn:
        df.to_pickle(r"./output/pickle")

    # Text File
    txtBtn = exportBtns.button("Export to Text File")
    if txtBtn:
        with open(r"./output/txtTest.txt", "w") as f:
            f.write(df.to_string(header=False, index=False))
