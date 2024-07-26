from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy
import cassandra
import uuid
from book_data import book_metadata, load_pdf, load_docx_doc, load_txt_doc, load_html_doc, load_epub_doc, load_doc, load_markdown
import time
import threading

# set keyspace names
REF_KEYSPACE = 'reference'
CONVO_KEYSPACE = 'conversations'

# set number of retries
MAX_RETRIES = 5

# get a session to the database
def get_session(host, port):
    cluster = Cluster(contact_points=[host], port=port, load_balancing_policy=DCAwareRoundRobinPolicy(), protocol_version=4)

    try:
        session = cluster.connect()
        # print(f'Connected to Cassandra host {host}:{port}')
    except Exception as e:
        print(f'Could not connect to Cassandra instance at {host}:{port}')
        print(e)
        return None

    return session

# execute a query on the database and retry if it fails
def execute_query(session, query):
    retries = 0
    try:
        for row in session.execute(query):
            print(f'Execute {query}: {row}')
    except Exception as e:
        print(e)
        if retries <= MAX_RETRIES:
            print('Retrying...')
            retries += 1
            time.sleep(5 * retries)
            execute_query(session, query)
        elif retries == MAX_RETRIES:
            print(f'Max retries reached, stopping execution. Please check to ensure the database is available. Error: {e}')
            print(f'Query: {query}')

def connect_database(host, port):
    session = get_session(host, port)
    return session

# create the reference and conversation keyspaces
def init_keyspaces(session):
    # create reference keyspace
    execute_query(session, f"CREATE KEYSPACE IF NOT EXISTS {REF_KEYSPACE} WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': '1'}}")
    # create conversations keyspace
    execute_query(session, f"CREATE KEYSPACE IF NOT EXISTS {CONVO_KEYSPACE} WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': '1'}}")

# create conversation table for Solr
def create_search_conversation(session, convo_name):
    # print('Creating solr conversation schema')
    # create conversations table
    execute_query(session, f"CREATE TABLE IF NOT EXISTS {CONVO_KEYSPACE}.{convo_name} (id UUID PRIMARY KEY, datetime timestamp, summary text, prompt text, response text)")
    # create index on responses
    execute_query(session, "CREATE SEARCH INDEX IF NOT EXISTS ON " + CONVO_KEYSPACE + "." + convo_name + " WITH COLUMNS prompt { indexed:true }, response { indexed:true }, summary { indexed:true }")
    # empty the conversations table
    execute_query(session, f"TRUNCATE TABLE {CONVO_KEYSPACE}.{convo_name}")
    # print('Created solr conversation schema')

# create a reference table for Solr
def create_search_agent(session, table_name):
    # print('Creating solr schema')
    # create books table
    execute_query(session, f"CREATE TABLE IF NOT EXISTS {REF_KEYSPACE}.{table_name} (id UUID, title text, author text, chapter int, chapter_title text, page_number int, chunk_number int, chunk_text text, subject text, PRIMARY KEY (id, page_number)) WITH CLUSTERING ORDER BY (page_number DESC)")
    # create index on title, text, and tags
    execute_query(session, "CREATE SEARCH INDEX IF NOT EXISTS ON " + REF_KEYSPACE + "." + table_name + " WITH COLUMNS title { indexed:false }, author { indexed:false }, chapter_title { indexed:false }, chunk_text { indexed:true }, subject { indexed:true }")
    # empty the books table
    execute_query(session, f"TRUNCATE TABLE {REF_KEYSPACE}.{table_name}")
    # print('Created solr schema')

# create a conversation table for vector search
def create_vector_conversation(session, convo_name):
    # print('Creating vector conversation schema')
    # create conversations table
    execute_query(session, f"CREATE TABLE IF NOT EXISTS {CONVO_KEYSPACE}.{convo_name} (id UUID PRIMARY KEY, datetime timestamp, summary text, prompt text, prompt_vector VECTOR <FLOAT, 384>)")
    # create index on prompt_vector
    execute_query(session, f"CREATE CUSTOM INDEX IF NOT EXISTS ann_index ON {CONVO_KEYSPACE}.{convo_name}(prompt_vector) USING 'StorageAttachedIndex'")
    # empty the conversations table
    execute_query(session, f"TRUNCATE TABLE {CONVO_KEYSPACE}.{convo_name}")
    # print('Created vector conversation schema')

# create a reference table for vector search (currently just using for embeddings storage)
def create_vector_agent(session, table_name):
    # print('Creating vector schema')
    # create books table
    execute_query(session, f"CREATE TABLE IF NOT EXISTS {REF_KEYSPACE}.{table_name} (id UUID, title text, page_number int, chunk_number int, chunk_vector VECTOR <FLOAT, 384>, PRIMARY KEY (id, page_number)) WITH CLUSTERING ORDER BY (page_number DESC)")
    # create index on page_vector
    execute_query(session, f"CREATE CUSTOM INDEX IF NOT EXISTS ann_index ON {REF_KEYSPACE}.{table_name}(chunk_vector) USING 'StorageAttachedIndex'")
    # empty the books table
    execute_query(session, f"TRUNCATE TABLE {REF_KEYSPACE}.{table_name}")
    # print('Created vector schema')

# insert query for each indiviudal chunk
def insert_query(session, query, values):
    retries = 0
    try:
        session.execute(query, values)
    except (cassandra.InvalidRequest, cassandra.Unauthorized, cassandra.AuthenticationFailed) as e:
        print(f'Error while executing query: {e}')
        print(f'Query: {query}')
        exit()
    except Exception as e:
        if retries < MAX_RETRIES:
            print(f'Retrying {retries} of {MAX_RETRIES}')
            time.sleep(5 * retries)
            retries += 1
            insert_query(session, query, values)
        elif retries == MAX_RETRIES:
            print(f'Max retries reached, stopping execution. Please check to ensure the database is available. Error: {e}')
            exit(1)

# insert a book into the database
def insert_book(title, file, author, solr_session, vector_session, table_name, subject):

    print(f'Loading: {title}')

    if file.lower().endswith('.html'):
        book = load_html_doc(file, title, author, subject)
    elif file.lower().endswith('.pdf'):
        book = load_pdf(file, title, author, subject)
    elif file.lower().endswith('.epub'):
        book = load_epub_doc(file, title, author, subject)
    elif file.lower().endswith('.txt'):
        book = load_txt_doc(file, title, author, subject)
    elif file.lower().endswith('.docx'):
        book = load_docx_doc(file, title, author, subject)
    elif file.lower().endswith('.doc'):
        book = load_doc(file, title, author, subject)
    elif file.lower().endswith('.md'):
        book = load_markdown(file, title, author, subject)

    # strip trailing spaces from title
    title = title.strip()

    try:
        chapters = book['chapters']
    # skip inserting if there's an error processing the file
    except UnboundLocalError as e:
        print(f"Error loading {file}. Error: {e}")
        exit(1)

    # set keyspace
    solr_session.execute(f"USE {REF_KEYSPACE}")
    vector_session.execute(f"USE {REF_KEYSPACE}")

    print(f'Inserting: {title}')

    # data looks like this:
    # {
    #    'author': 'Games Workshop',
    #    'title': 'Warhammer 40,000 10th Edition Core Rules',
    #    'chapters'  : {
    #         1: {
    #             'title': 'Rules Key',
    #             'pages': {
    #                 1: {
    #                     'subject': 'subject',
    #                     'chunks': {
    #                         0: {
    #                             'text': 'text',
    #                             'embedding': [1, 2, 3]
    #                         },
    #                         1: {
    #                             'text': 'text',
    #                             'embedding': [1, 2, 3]
    #                         }
    #                     }
    #                 }
    #             }
    #         }
    #     }
    # }

    for chapter in chapters:
        for page in chapters[chapter]['pages']:
            for chunk in chapters[chapter]['pages'][page]['chunks']:
                # generate a uuid for the record
                id = uuid.uuid4()

                # insert the chunk into the solr table
                solr_insert = f"INSERT INTO {table_name}"
                solr_insert += """(id, title, author, chapter, chapter_title, page_number, chunk_number, chunk_text, subject)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                insert_query(solr_session, solr_insert, (id, title, author, int(chapter), chapters[chapter]['title'], int(page), int(chunk), chapters[chapter]['pages'][page]['chunks'][chunk]['text'], subject))

                # insert the chunk into the vector table
                vector_insert = f"INSERT INTO {table_name}"
                vector_insert += """(id, title, page_number, chunk_number, chunk_vector)
                VALUES (%s, %s, %s, %s, %s)"""
                insert_query(vector_session, vector_insert, (id, title, int(page), int(chunk), chapters[chapter]['pages'][page]['chunks'][chunk]['embedding'].tolist()))

# threading function to load books
def book_thread(book, metadata, solr_session, vector_session, table_name, agent_context):
    insert_book(book, metadata[book]['file'], metadata[book]['author'], solr_session, vector_session, table_name, agent_context)

# function to insert books into the database
def insert_books(solr_session, vector_session, table_name, agent_context, files):
    metadata = book_metadata(files, agent_context)

    for book in metadata:
        t = threading.Thread(target=book_thread, args=(book, metadata, solr_session, vector_session, table_name, agent_context))
        t.start()
        t.join()
        print(f'Finished loading {book}')

    print('Loading sources complete.')

# create the keyspaces in the databases
def create_keyspaces(solr_session, vector_session):
    init_keyspaces(solr_session)
    init_keyspaces(vector_session)

# create the searchable conversation tables
def create_new_conversation(convo_name, solr_session, vector_session):
    create_search_conversation(solr_session, convo_name)
    create_vector_conversation(vector_session, convo_name)

# check if the keyspaces exist
def db_checks(session):
    ref_keyspace_exists = False
    convo_keyspace_exists = False

    try:
        rows = session.execute("SELECT keyspace_name FROM system_schema.keyspaces")
        for row in rows:
            if row.keyspace_name == REF_KEYSPACE:
                ref_keyspace_exists = True
            if row.keyspace_name == CONVO_KEYSPACE:
                convo_keyspace_exists = True

    except Exception as e:
        print('Error checking keyspaces')
        print(e)
    
    return ref_keyspace_exists, convo_keyspace_exists

# check if there are any tables in the keyspaces
def books_checks(session, table_name):
    table_exists = False

    try:
        rows = session.execute(f"SELECT table_name FROM system_schema.tables WHERE keyspace_name = '{REF_KEYSPACE}'")
        for row in rows:
            if row.table_name == table_name:
                table_exists = True

    except Exception as e:
        print('No books tables found')

    return table_exists

# list the tables in the keyspaces
def list_tables(session, keyspace):
    tables = []

    try:
        rows = session.execute(f"SELECT table_name FROM system_schema.tables WHERE keyspace_name = '{keyspace}'")
        for row in rows:
            tables.append(row.table_name)

    except Exception as e:
        print('No tables found')

    return tables

# list the agent contexts from the tables
def list_agent_contexts(session, keyspace):
    tables = list_tables(session, keyspace)

    contexts = []
    # for each table, list the subject from the first row
    for table in tables:
        rows = session.execute(f"SELECT subject FROM {keyspace}.{table} LIMIT 1")
        for row in rows:
            contexts.append(row.subject)

    return contexts