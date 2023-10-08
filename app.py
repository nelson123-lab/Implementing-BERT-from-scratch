import streamlit as st
from dotenv import load_dotenv
from pyPDF2 import PdfReader

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # Inorder to extract each page contents.
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def main():
    load_dotenv()
    st.set_page_config(page_title = "Chat with Jake", page_icon=":books")

    st.header("Chat with Jake")
    st.text_input('What should I guide you with?')

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your documents", accept_multiple_files = True)
        if st.button("Process"):
            # To show a processing circle instead of staying still.
            with st.spinner("Processing"):
                # Get pdf
                raw_text = get_pdf_text(pdf_docs)
                # get the text chunks

                # create vector store.
if __name__ == '__main__':
    main()