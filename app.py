import streamlit as st
import pdfplumber
import requests
import csv
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to download PDF from a URL
def download_pdf(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    extracted_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            extracted_text.append([i, text])
    return extracted_text

# Function to save extracted text to a CSV file
def save_to_csv(extracted_text, csv_path):
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Page Number', 'Extracted Text'])
        writer.writerows(extracted_text)

# Function to calculate TF, IDF, and TF-IDF and save to a CSV file
def calculate_tf_idf(csv_path, tf_idf_csv_path):
    df = pd.read_csv(csv_path)
    
    # Using TfidfVectorizer to calculate TF, IDF, and TF-IDF
    vectorizer = TfidfVectorizer(use_idf=True, norm=None)  # norm=None to get raw term frequencies
    tfidf_matrix = vectorizer.fit_transform(df['Extracted Text'].fillna(''))
    
    # Extracting TF (raw term counts)
    tf = tfidf_matrix.toarray()
    
    # Calculating IDF
    idf_scores = vectorizer.idf_
    
    # Extracting terms
    terms = vectorizer.get_feature_names_out()
    
    # Calculating term occurrence in documents
    doc_occurrences = (tf > 0).sum(axis=0)
    
    # Combining TF, IDF, and TF-IDF into a DataFrame
    tf_df = pd.DataFrame(tf, columns=terms)
    idf_df = pd.DataFrame(idf_scores.reshape(1, -1), columns=terms)
    doc_occurrences_df = pd.DataFrame(doc_occurrences.reshape(1, -1), columns=terms)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=terms)
    
    # Save results to CSV
    with pd.ExcelWriter(tf_idf_csv_path) as writer:
        tf_df.to_excel(writer, sheet_name="TF", index=False)
        idf_df.to_excel(writer, sheet_name="IDF", index=False)
        tfidf_df.to_excel(writer, sheet_name="TF-IDF", index=False)
        doc_occurrences_df.to_excel(writer, sheet_name="Term Occurrences", index=False)

# Streamlit UI
st.set_page_config(page_title="PDF Text Extractor with TF, IDF, and TF-IDF", page_icon="📄")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { 
        background-color: #f8f9fa; 
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📄 PDF Text Extractor with TF, IDF, and TF-IDF Calculation")
st.subheader("Extract, Save PDF Text as CSV, and Calculate TF, IDF, and TF-IDF")
st.markdown("""
    This tool allows you to extract text from any PDF file by either providing its URL or uploading a file directly.
    The extracted text will be saved into a CSV file, which you can then use to calculate Term Frequency (TF), 
    Inverse Document Frequency (IDF), and TF-IDF. You can also check how many documents contain each term.
""")

# User option to choose URL or file upload
option = st.radio("Choose the source of the PDF:", ('URL', 'File Upload'))

if option == 'URL':
    # Input for PDF URL
    pdf_url = st.text_input("🔗 Enter the URL of the PDF file:", placeholder="https://example.com/yourfile.pdf")

    # Button to trigger the extraction process
    if st.button("Extract Text from URL"):
        if pdf_url:
            try:
                # Define paths
                pdf_file = "downloaded.pdf"
                csv_file = "extracted_text.csv"
                
                # Display progress
                with st.spinner('Downloading PDF...'):
                    download_pdf(pdf_url, pdf_file)
                    st.success("PDF downloaded successfully!")
                
                with st.spinner('Extracting text...'):
                    # Extract text from the PDF
                    extracted_text = extract_text_from_pdf(pdf_file)
                    st.success("Text extracted successfully!")
                
                with st.spinner('Saving to CSV...'):
                    # Save the extracted text to a CSV
                    save_to_csv(extracted_text, csv_file)
                    st.success(f"Extracted text saved to {csv_file}")
                
                # Provide download link for the CSV file
                with open(csv_file, "rb") as file:
                    st.download_button(
                        label="📥 Download Extracted Text as CSV",
                        data=file,
                        file_name=csv_file,
                        mime="text/csv",
                        help="Click to download the extracted text as a CSV file"
                    )
                
                # Cleanup
                os.remove(pdf_file)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a valid URL.")

elif option == 'File Upload':
    # File uploader for PDF
    uploaded_file = st.file_uploader("📂 Upload your PDF file", type=["pdf"])

    # Button to trigger the extraction process
    if st.button("Extract Text from File"):
        if uploaded_file is not None:
            try:
                # Define paths
                pdf_file = "uploaded.pdf"
                csv_file = "extracted_text.csv"
                
                # Save the uploaded file temporarily
                with open(pdf_file, "wb") as file:
                    file.write(uploaded_file.getbuffer())
                
                st.success("PDF file uploaded successfully!")
                
                with st.spinner('Extracting text...'):
                    # Extract text from the PDF
                    extracted_text = extract_text_from_pdf(pdf_file)
                    st.success("Text extracted successfully!")
                
                with st.spinner('Saving to CSV...'):
                    # Save the extracted text to a CSV
                    save_to_csv(extracted_text, csv_file)
                    st.success(f"Extracted text saved to {csv_file}")
                
                # Provide download link for the CSV file
                with open(csv_file, "rb") as file:
                    st.download_button(
                        label="📥 Download Extracted Text as CSV",
                        data=file,
                        file_name=csv_file,
                        mime="text/csv",
                        help="Click to download the extracted text as a CSV file"
                    )
                
                # Cleanup
                os.remove(pdf_file)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload a PDF file.")

# Section to upload a CSV file and calculate TF, IDF, and TF-IDF
st.subheader("Calculate TF, IDF, and TF-IDF from Uploaded CSV")

uploaded_csv = st.file_uploader("📂 Upload your CSV file with extracted text", type=["csv"])

if uploaded_csv is not None:
    try:
        tf_idf_csv_file = "tf_idf_scores.xlsx"
        
        # Save the uploaded CSV file temporarily
        with open("uploaded_text.csv", "wb") as file:
            file.write(uploaded_csv.getbuffer())
        
        # Calculate TF, IDF, and TF-IDF and save to a CSV
        calculate_tf_idf("uploaded_text.csv", tf_idf_csv_file)
        st.success(f"TF, IDF, and TF-IDF scores calculated and saved to {tf_idf_csv_file}")
        
        # Provide download link for the TF-IDF Excel file
        with open(tf_idf_csv_file, "rb") as file:
            st.download_button(
                label="📥 Download TF, IDF, and TF-IDF Scores as Excel",
                data=file,
                file_name=tf_idf_csv_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Click to download the TF, IDF, and TF-IDF scores as an Excel file"
            )
        
        # Cleanup
        os.remove("uploaded_text.csv")
        os.remove(tf_idf_csv_file)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
