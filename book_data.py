from sentence_transformers import SentenceTransformer
import torch.cuda as cuda
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import docx2txt
import ebooklib
from ebooklib import epub
import yaml
import textract
import markdown

# load the vector embeddings model
model = SentenceTransformer(
    'all-MiniLM-L6-v2',
    device='cuda' if cuda.is_available() else 'cpu'
    )

# split a page of text into chunks
def text_chunks(text, chunk_size):
    chunks = {}
    for i in range(0, len(text), chunk_size):
        chunk_text = text[i:i + chunk_size]
        chunks[i] = {
            'text': chunk_text,
            'embedding': model.encode(chunk_text)
        }
    return chunks

# load a book from a pdf file
def load_pdf(book, title, author, subject):

    book = PdfReader(book)

    data = {
        'title': title,
        'author': author,
        'num_pages': len(book.pages),
        'chapters': {
            0: {
                'title': 'None',
                'pages': {}
            }
        }
    }

    for page in range(len(book.pages)):
        text = book.pages[page].extract_text()
        data['chapters'][0]['pages'][page] = {
            'subject': subject,
            'chunks': text_chunks(text, 1000)
        }

    return data

# load a book from a html file
def load_html_doc(html, title, author, subject):
    # Open the HTML file
    with open(html, "r") as file:
        # Read the contents of the file
        html_content = file.read()

    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract the plain text from the HTML
    text = soup.get_text()

    data = {
        'title': title,
        'author': author,
        'num_pages': 1,
        'chapters': {
            0: {
                'title': 'None',
                'pages': {
                    0: {
                        'subject': subject,
                        'chunks': text_chunks(text, 1000)
                    }
                }
            }
        }
    }
    
    return data

# load a book from an epub file
def load_epub_doc(file, title, author, subject):
    book = epub.read_epub(file)

    chapter_number = 1

    data = {
        'title': title,
        'author': author,
        'num_pages': len(list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))),
        'chapters': {}
    }

    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    for item in items:
        if 'chapter' in item.get_name().lower():
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = [para.get_text() for para in soup.find_all('p')]
            chapter = item.get_name()
            chapter_number += 1
            data['chapters'][chapter_number] = {
                'title': chapter,
                'pages': {
                    0: {
                        'subject': subject,
                        'chunks': text_chunks(text, 1000)
                    }
                }
            }

    return data

# load a book from a txt file
def load_txt_doc(file, title, author, subject):
    with open(file, 'r') as f:
        text = f.read()

    data = {
        'title': title,
        'author': author,
        'num_pages': 1,
        'chapters': {
            0: {
                'title': 'None',
                'pages': {
                    0: {
                        'subject': subject,
                        'chunks': text_chunks(text, 1000)
                    }
                }
            }
        }
    }

    return data

# load a book from a docx file
def load_docx_doc(file, title, author, subject):
    text = docx2txt.process(file)

    data = {
        'title': title,
        'author': author,
        'num_pages': 1,
        'chapters': {
            0: {
                'title': 'None',
                'pages': {
                    0: {
                        'subject': subject,
                        'chunks': text_chunks(text, 1000)
                    }
                }
            }
        }
    }

    return data

# load a book from a doc file
def load_doc(file, title, author, subject):
    text = textract.process(file)

    data = {
        'title': title,
        'author': author,
        'num_pages': 1,
        'chapters': {
            0: {
                'title': 'None',
                'pages': {
                    0: {
                        'subject': subject,
                        'chunks': text_chunks(text, 1000)
                    }
                }
            }
        }
    }

    return data

# load a book from a markdown file
def load_markdown(file, title, author, subject):
    # open the file and read the contents
    with open(file, 'r') as f:
        contents = f.read()

    # parse as html
    html = markdown.markdown(contents)

    # convert to soup
    soup = BeautifulSoup(html, 'html.parser')

    # extract the plain text
    text = soup.get_text()

    data = {
        'title': title,
        'author': author,
        'num_pages': 1,
        'chapters': {
            0: {
                'title': 'None',
                'pages': {
                    0: {
                        'subject': subject,
                        'chunks': text_chunks(text, 1000)
                    }
                }
            }
        }
    }

    return data

def load_config():
    # load the config file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def write_config(config):
    # write the config file
    with open('config.yaml', 'w') as file:
        yaml.dump(config, file)

# try to locate an author in the filename
def split_author(file, extension):

    path = file.split(extension)[0]
    filename = path.split('/')[-1]

    if 'author ' in filename.lower():
        try:
            author = filename.split('Author ')[1]
            title = filename.split('Author ')[0]
        except:
            author = filename.split('author ')[1]
            title = filename.split('author ')[0]
    else:
        author = 'Unknown'
        title = filename

    return author, title

# function to get the metadata of the books
def book_metadata(files, context):

    book_metadata = {}

    print(f'Found {len(files)} files')

    # loop through the files
    for file in files:

        if file.lower().endswith('.html'):
            author, title = split_author(file, '.html')
            pages = 1

        elif file.lower().endswith('.pdf'):
            # open the pdf file and store the title, author, and number of pages
            book = PdfReader(file)
            pages = len(book.pages)
            author, title = split_author(file, '.pdf')

        elif file.lower().endswith('.epub'):
            book = epub.read_epub(file)
            pages = len([book for book in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)])
            title = book.get_metadata('DC', 'title')[0][0]
            author = book.get_metadata('DC', 'creator')[0][0]

        elif file.lower().endswith('.txt'):
            pages = 1
            author, title = split_author(file, '.txt')

        elif file.lower().endswith('.docx'):
            pages = 1
            author, title = split_author(file, '.docx')

        elif file.lower().endswith('.doc'):
            pages = 1
            author, title = split_author(file, '.doc')

        elif file.lower().endswith('.md'):
            author, title = split_author(file, '.md')
            pages = 1

        else:
            print(f'Skipping {file} as it is not a recognized file type')
            continue

        # store the metadata in a dict
        book_metadata[title] = {
            'author': author,
            'pages': pages,
            'subject': context,
            'file': file,
        }

    return book_metadata