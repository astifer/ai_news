import streamlit as st
import os
import io
import logging

import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

import streamlit as st

st.title('AI news')
st.write('Upload file and then we detect duplicates and return unique news with categories')

csv_url = "http://localhost:8000/csv"
json_url = "http://localhost:8000/json"
excel_url = "http://localhost:8000/excel"



def process(file, content_type, server_url: str):
    m = MultipartEncoder(fields={"file": (file.name, file, content_type)})

    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )
 
    return r.content

option = st.selectbox(  
    'How would you like to upload file?',
    ('Excel', 'Csv',  'Json'))

st.write('You selected:', option)

if option:
    if option == 'Csv':
        input_csv = st.file_uploader("insert csv", type=['csv'])  
        
        if input_csv:
            compute_button = st.button('Compute')
        
            if compute_button:
                    file_bytes = process(input_csv, "text/csv", csv_url)
                    st.download_button(label='Download csv', data=file_bytes, file_name='file.csv', mime='text/csv')

    elif option == 'Excel':
        
        input_excel = st.file_uploader("insert excel",type=["xlsx", "xls"]) 

        if input_excel:
            compute_button = st.button('Compute')
            
            if compute_button:
                st.write('Process...')
                processed_data_bytes = process(input_excel, "text/excel", excel_url)
                st.download_button(label='Download excel', data=processed_data_bytes, file_name='processed_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    elif option == 'Json':
        
        input_json = st.file_uploader("insert json") 
        if input_json:
            compute_button = st.button('Compute')
        
            if compute_button:
                file_bytes = process(input_json, "application/json", json_url)
                st.download_button(label='Download json', data=file_bytes, file_name='file.json', mime='text/csv')

    



    

