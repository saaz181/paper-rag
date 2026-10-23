import os
from pypdf import PdfReader
import json


def extract_pdf(pdf_file):
    pages = []
    reader = PdfReader(pdf_file)

    for page_number, page in enumerate(reader.pages, start=1):

        pages.append( {'page': page_number, 'text': page.extract_text() or "" } )

    metadata = { }

    if reader.metadata:
        metadata = {
            str(key): str(value) for key, value in reader.metadata.items()
        }

    return {
        'pages': pages,
        'metadata': metadata
    }



def ingest_folder(data_folder):
    documents = {}

    file_names = list(filter( lambda f: f.endswith('.pdf'), os.listdir(data_folder)))

    for fn in file_names:
        file = os.path.join(data_folder, fn)

        documents[fn] = extract_pdf(file)

    return documents




if __name__ == "__main__":


    parent_folder = os.getcwd()
    data_folder = os.path.join(parent_folder, 'data')

    extracted_documents = ingest_folder(data_folder)


    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(extracted_documents, f, indent=2, ensure_ascii=False)



    
