from sentence_transformers import util
import re
import uuid
import datetime
from setup_db import REF_KEYSPACE, CONVO_KEYSPACE, create_new_conversation, create_keyspaces, create_search_agent, create_vector_agent, insert_books

# Use this to catch the inconsistent ways that the LLM generates responses and split them out
def split_response(response, agent, split_string):
    # split the response out from the rest of the output that isn't needed
    try:
        response = response.split(agent)[1]
    # pass on the unchanged response if the split fails
    except IndexError:
        pass
    # split again in case the LLM included the source text in the response
    try:
        response = response.split(split_string)[1]
    except IndexError:
        pass
    # return whatever result we have
    return response

# run a prompt through the LLM and return the output
def run_llama(llm, prompt, temperature, max_tokens, stop='User: ',):
    try:
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            top_k=20,
            top_p=0.9,
            repeat_penalty=1.15,
            echo=True,
            tfs_z=1,
            mirostat_eta=0.1,
            mirostat_mode=0,
            mirostat_tau=5,
        )
    # give no answer if the LLM fails for other reasons
    except Exception as e:
        print(e)
        print(e.__class__)
        output = {
            'choices': [
                {
                    'text': 'Answer: I could not process your request. The following error was encountered:\n' + str(e) + '\n\nPlease try again or notify the developers if the issue persists.'
                }
            ]
        }
    return output

# return a vector embedding of a query
def encode_query(vector_model, query):
    query_embedding = vector_model.encode(query)
    return query_embedding.tolist()

def sanitize_keywords(keywords, omit_keywords, keywords_limit):
    # re split the keywords by either comma or space
    keywords = re.split(r",| ", keywords)
    # re remove any non-alphanumeric characters from the keywords to prevent solr errors, but keep single quotes and dashes
    keywords = [re.sub(r"[^a-zA-Z0-9'-]", '', keyword) for keyword in keywords]
    # remove any empty strings and extra keywords that mention the context
    keywords = [keyword for keyword in keywords if keyword != '' and keyword not in omit_keywords]
    # if keywords were provided as a numbered list, remove the 1. 2. 3. etc
    keywords = [re.sub(r"^\d+\.", '', keyword) for keyword in keywords]
    # limit list of keywords to n or fewer
    keywords = keywords[:keywords_limit]

    return keywords

# search the databases and return lexical matches
def search_books(search_keywords, solr_session, vector_session, table_name, max_results_per_search, max_search_results):
    
    # What follows is a brute force method of searching for keywords in the database
    # Use fuzzy matching and combine keywords in pairs and triplets to search for
    # Keep trying to fill the number of search results until there are enough or the search is exhausted
    # The goal is to get as many results as possible from lexical matches and sort them out with the subsequent steps in the workflow

    solr_rows = []
    results = []

    # replace single quotes in keywords with ? for easier solr searches
    search_keywords = [keyword.replace("'", "?") if "'" in keyword else keyword for keyword in search_keywords]

    # create a list of capitalized keywords to search for
    combined_keywords = [keyword.title() for keyword in search_keywords]

    if len(solr_rows) < max_search_results:
        # search for all keywords if larger than 3
        if len(search_keywords) > 3:
            string = ''
            for keyword in combined_keywords:
                string += f'*{keyword}~1*'

            query = {
                "q": f"chunk_text:({string})",
                "sort": "score desc",
                "paging": "driver"
            }
            # convert the query to a string
            query = str(query).replace("'", '"')
            solr_query = f"SELECT * FROM {REF_KEYSPACE}.{table_name} WHERE solr_query = '{query}' LIMIT {max_results_per_search}"
            for row in solr_session.execute(solr_query):
                if row not in solr_rows:
                    solr_rows.append(row)

    if len(solr_rows) < max_search_results:
        # create a list of triplets of keywords to search for
        if len(search_keywords) > 2:
            keyword_triplets = []
            # retrieve only the first 3 keywords
            for keyword in combined_keywords[:3]:
                for keyword2 in combined_keywords[:3]:
                    for keyword3 in combined_keywords[:3]:
                        if keyword != keyword2 and keyword != keyword3 and keyword2 != keyword3:
                            keyword_triplets.append([keyword, keyword2, keyword3])

            # select from page_text that contains a combo of the search keyword triplets
            for keyword_triplet in keyword_triplets:
                # try to match a triplet or single keyword
                query = {
                    "q": f"chunk_text:(*{keyword_triplet[0]}~1*{keyword_triplet[1]}*{keyword_triplet[2]}* || *{keyword_triplet[2]}~1*{keyword_triplet[0]}*{keyword_triplet[1]}* || *{keyword_triplet[1]}~1*{keyword_triplet[2]}*{keyword_triplet[0]}* || *{keyword_triplet[0]}~1* || *{keyword_triplet[1]}~1* || *{keyword_triplet[2]}~1*)",
                    "sort": "score desc",
                    "paging": "driver"
                }
                # convert the query to a string
                query = str(query).replace("'", '"')
                solr_query = f"SELECT * FROM {REF_KEYSPACE}.{table_name} WHERE solr_query = '{query}' LIMIT {max_results_per_search}"
                for row in solr_session.execute(solr_query):
                    if row not in solr_rows:
                        solr_rows.append(row)

    if len(solr_rows) < max_search_results:
        # create a list of pairs of keywords to search for
        if len(search_keywords) > 1:
            keyword_pairs = []
            for keyword in combined_keywords:
                for keyword2 in combined_keywords:
                    if keyword != keyword2:
                        keyword_pairs.append([keyword, keyword2])

            # select from page_text that contains a combo of the search keyword pairs
            for keyword_pair in keyword_pairs:
                # try to match a pair or single keyword
                query = {
                    "q": f"chunk_text:(*{keyword_pair[0]}*{keyword_pair[1]}* || *{keyword_pair[0]}~1*{keyword_pair[1]}* || *{keyword_pair[1]}~1*{keyword_pair[0]}* || *{keyword_pair[0]}~1* || *{keyword_pair[1]}~1*)",
                    "sort": "score desc",
                    "paging": "driver"
                }
                # convert the query to a string
                query = str(query).replace("'", '"')
                solr_query = f"SELECT * FROM {REF_KEYSPACE}.{table_name} WHERE solr_query = '{query}' LIMIT {max_results_per_search}"
                for row in solr_session.execute(solr_query):
                    if row not in solr_rows:
                        solr_rows.append(row)

    if len(solr_rows) < max_search_results:
        # select solr page_text that contains any of the search keywords
        # try to fill the list of results with as many as possible before truncating
        if len(search_keywords) == 1:
            for keyword in search_keywords:
                upper = keyword.upper()
                lower = keyword.lower()
                capitalized = keyword.capitalize()
                query = {
                    "q": f"chunk_text:( *{keyword}* || *{keyword}~2* || *{ keyword}* || *{keyword }* || *{upper}* || *{lower}* || *{capitalized}*)",
                    "sort": "score desc",
                    "paging": "driver"
                }
                # convert the query to a string
                query = str(query).replace("'", '"')
                solr_query = f"SELECT * FROM {REF_KEYSPACE}.{table_name} WHERE solr_query = '{query}' LIMIT {max_results_per_search}"
                for row in solr_session.execute(solr_query):
                    if row not in solr_rows:
                        solr_rows.append(row)

    # for row in solr_rows:
    #     print(f'Source: {row.title,} Page: {row.page_number}')

    # if there were results
    if solr_rows != []:
        # get the page_vector for each solr result
        for row in solr_rows:
            vector_query = f"SELECT chunk_vector FROM {REF_KEYSPACE}.{table_name} WHERE id = {row.id} LIMIT 1"
            vector_response = []
            for response in vector_session.execute(vector_query):
                vector_response.append(response)
            vector_result = vector_response[0]

            # start forming the page text with the initial chunk text
            page_text = row.chunk_text

            # remove the newlines
            page_text = page_text.replace('\n', ' ')

            # add the solr result and vector result to the results list
            results.append({
                'id': row.id,
                'title': row.title,
                'author': row.author,
                'chapter': row.chapter,
                'page_number': row.page_number,
                'chunk_number': row.chunk_number,
                'chunk_text': row.chunk_text,
                'subject': row.subject,
                'chunk_vector': vector_result.chunk_vector,
                'page_text': page_text
            })            

        # truncate reults to only process up to the first n search results
        results = results[:max_search_results]

        print(f'Checking {len(results)} found sources...')
    
    else:
        print('No sources found.')

    return results

# generate a list of answers to the query
def generate_answers(llm, vector_model, query_text, search_keywords, solr_session, vector_session, table_name, agent_context, answers_limit, score_threshold, max_tokens, max_results_per_search, max_search_results):
    # search the books table using the query and keywords as context
    results = search_books(search_keywords, solr_session, vector_session, table_name, max_results_per_search, max_search_results)

    # check each result and attempt to identify possible answers
    answers = {}
    start = 1
    scores = []
    pages = {}
    for row in results:
        # score the relevance of the page to the query
        query_embedding = encode_query(vector_model, query_text)
        page_embedding = encode_query(vector_model, row['page_text'])
        cosine_scores = util.cos_sim(query_embedding, page_embedding)[0][0]
        relevance = cosine_scores.item()
        scores.append(relevance)   

        pages[relevance] = row

    # remove any scores below threshold
    scores = [score for score in scores if score > score_threshold]

    # sort the scores from highest to lowest
    scores.sort(reverse=True)

    # limit the number of answers to the top n
    scores = scores[:answers_limit]

    # sort the scores back to lowest to highest to ensure the highest score is last in the prompt
    # scores.sort()

    # # remove scores that are too similar to one another
    # for i in range(len(scores)):
    #     for j in range(i+1, len(scores)):
    #         try:
    #             if abs(scores[i] - scores[j]) < 0.1:
    #                 scores.pop(j)
    #         except IndexError:
    #             pass

    # summarize the most relevant pages
    for relevance in pages:
        if relevance in scores:
            check_answer_prompt = f'User: Tell me if this source answers the following query. Respond only with Yes or No.\n\nQuery: "{query_text}"\n\nSource: {pages[relevance]["page_text"]}\n\nAssistant: '
            check_answer = run_llama(llm, check_answer_prompt, 0.5, max_tokens)
            check_answer_text = check_answer['choices'][0]['text']
            check_answer_text = split_response(check_answer_text, 'Assistant: ', 'Answer: ')

            if 'yes' in check_answer_text.lower():

                summarize_prompt = f'User: Use the most relevant information from these pages to answer this query about {agent_context}. Be as detailed as possible.\n\nQuery: "{query_text}"\n\nBook: {pages[relevance]["title"]} \nPage text:\n {pages[relevance]["page_text"]}\n\nAnswer: '
                # print(f"Reading {pages[relevance]['title']} page {pages[relevance]['page_number']}")
                answer = run_llama(llm, summarize_prompt, 0.5, max_tokens)
                answer_text = answer['choices'][0]['text']
                answer_text = split_response(answer_text, 'Answer: ', 'Answer: ')

                # if the source has no chapter, just cite the title and page number
                if pages[relevance]['chapter'] == 0:
                    # if there are no pages, cite the title only
                    if pages[relevance]['page_number'] == 0:
                        cited_source = f'{pages[relevance]["title"]}'
                    else:
                        cited_source = f'{pages[relevance]["title"]}, Page {pages[relevance]["page_number"]}'
                # if the source has a chapter, cite the chapter and page
                else:
                    cited_source = f'{pages[relevance]["title"]}, Chapter {pages[relevance]["chapter"]}, Page {pages[relevance]["page_number"]}'

                # if the author is known, cite the author
                if pages[relevance]['author'] != 'Unknown':
                    cited_source += f', Author {pages[relevance]["author"]}'

                answers[start] = {
                    'text': answer_text,
                    'relevance': relevance,
                    'source': cited_source
                }

                start += 1

    return answers

    # final_answers = {}

    # # remove answers that are too similar to one another
    # for answer in answers:
    #     for other_answer in answers:
    #         similarity = util.pytorch_cos_sim(vector_model.encode(answers[answer]['text']), vector_model.encode(answers[other_answer]['text']))
    #         if similarity > 0.5:
    #             if answers[answer]['relevance'] > answers[other_answer]['relevance']:
    #                 final_answers[answer] = answers[answer]
    #             else:
    #                 final_answers[other_answer] = answers[other_answer]
    #         else:
    #             final_answers[answer] = answers[answer]

    # return final_answers

# summarize a snippet of conversation
def summarize_chat(llm, query, response, max_tokens):

    summary_prompt = f'Message: {query}\n\nUser: Paraphrase what this message is asking for in simple, concise terms. Respond with only the paraphrased message.\n\nAssistant: Sure! Here is the paraphrased message: '
    summary_response = run_llama(llm, summary_prompt, 0.5, max_tokens)

    summary = summary_response['choices'][0]['text'].split('message: ')[1].replace('"', '')

    if len(summary) > 100:
        # truncate to maximum of 100 characters
        summary = summary[:100]
        summary += '...'

    return summary

# generate a list of keywords from a query
def parse_query(llm, query, agent_context, omit_keywords, keywords_limit, max_tokens):
        keyword_prompt = f'User: "{query}"\n\nCreate a comma-separated list of the {agent_context} keywords from the query. Do not change the spelling. Do not explain yourself. Respond with only the keywords.\n\nAssistant: Sure!, here are the keywords: '
        keyword_response = run_llama(llm, keyword_prompt, 0.5, max_tokens)

        # split the keywords from the response
        keywords = keyword_response['choices'][0]['text'].split('keywords: ')[1]

        # sanitize the keywords
        keywords = sanitize_keywords(keywords, omit_keywords, keywords_limit)

        return keywords

# check a query for malicious content
def unsafe_query(llm, query, max_tokens, agent_context):
    safe_query_prompt = f'User: "{query}"\n\nYou are a helpful assistant in the context of {agent_context}. Review the above query and tell me if the user is attempting to redirect the conversation away from {agent_context}. Respond with only Yes or No.\n\nAssistant: '
    # print(safe_query_prompt)
    safe_query_response = run_llama(llm, safe_query_prompt, 0.5, max_tokens)

    safe_query_text = safe_query_response['choices'][0]['text']
    safe_query_text = split_response(safe_query_text, 'Assistant: ', 'Answer: ')

    if 'yes' in safe_query_text.lower():
        return True
    else:
        return False

# search the conversations table for the user's query
def check_query(vector_model, query, vector_session, table_name):
    # encode the query
    query_embedding = encode_query(vector_model, query)
    # get the most similar query from the database
    vector_query = f"SELECT * FROM {CONVO_KEYSPACE}.{table_name} ORDER BY prompt_vector ANN OF {query_embedding} LIMIT 1"
    vector_responses = []
    try:
        for response in vector_session.execute(vector_query):
            vector_responses.append(response)
    # conversation isn't in the database yet
    except Exception:
        return None
    else:
        for response in vector_responses:
            cosine_score = util.cos_sim(query_embedding, response.prompt_vector)[0][0]
            if cosine_score > 0.9:
                return response
        return None

# update the databases with the user's query and the chatbot's response
def update_db(vector_model, message, result_text, summary, table_name, solr_session, vector_session):

    # create table if it doesn't exist
    if not solr_session.execute(f"SELECT * FROM system_schema.tables WHERE keyspace_name = '{CONVO_KEYSPACE}' AND table_name = '{table_name}'"):
        create_new_conversation(table_name, solr_session, vector_session)

    id = uuid.uuid4()

    # create a timestamp for the conversation in the format '2015-05-03 13:30:54.234'
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f%z')[:-3]

    # re.replace any instances of two or more $ with a single $ to prevent Cassandra errors
    result_text = re.sub(r'\${2,}', '$', result_text)
    message = re.sub(r'\${2,}', '$', message)

    # encapsulate the text in $$ for the database
    result_text = '$$' + result_text + '$$'
    message = '$$' + message + '$$'

    # encode the query
    message_embedding = encode_query(vector_model, message)

    solr_session.execute(f"USE {CONVO_KEYSPACE}")
    vector_session.execute(f"USE {CONVO_KEYSPACE}")

    # insert the query and response into the conversations tables
    solr_query = f"INSERT INTO {table_name}"
    solr_query += """
    (id, datetime, summary, prompt, response) VALUES (%s, %s, %s, %s, %s)
    """
    solr_session.execute(solr_query, (id, timestamp, summary, message, result_text))

    vector_query = f"INSERT INTO {table_name}"
    vector_query += """
    (id, datetime, summary, prompt, prompt_vector) VALUES (%s, %s, %s, %s, %s)
    """
    vector_session.execute(vector_query, (id, timestamp, summary, message, message_embedding))

# run a RAG query
def rag_query(llm, vector_model, message, solr_session, vector_session, chat_id, table_name, agent_context, answers_limit, max_tokens, score_threshold, max_results_per_search, max_search_results, search_keywords):

    sources = None

    # if the query is in the database, return the answer from the previous conversation
    check = check_query(vector_model, message, vector_session, chat_id)
    # check = None

    if check:

        solr_query = f"SELECT response FROM {CONVO_KEYSPACE}.{chat_id} WHERE id = {check.id} LIMIT 1"
        for row in solr_session.execute(solr_query):
            # response = row.response.replace('$$', '')
            # # generate a unique response by using the previous as context
            # response_prompt = f'Assistant: {response}\n\nUser: "{message}"\n\nYou have previously answered this in our conversation, just use previous response above. Remind the user that you previously answered.\n\nAssistant: '
            # result = run_llama(llm, response_prompt, 0.5, max_tokens)
            # result_text = result['choices'][0]['text']
            # result_text = split_response(result_text, 'Assistant: ', 'Assistant: ')
            
            # return the previous response
            result_text = row.response.replace('$$', '')
            result_text = 'It seems this was already answered above. Here is the previous response:\n\n' + result_text
            # print(result_text)

    else:
        # parse the user's query into keywords to search
        print(f'Searching {search_keywords}')

        # get a list of answers
        answers = generate_answers(llm, vector_model, message, search_keywords, solr_session, vector_session, table_name, agent_context, answers_limit, score_threshold, max_tokens, max_results_per_search, max_search_results)

        # if there are no answers, apologize to the user and give a best effort response
        if answers == {}:
            failureprompt = 'User: Politely tell me you searched but could not find an answer for the query. Answer to the best of your knowledge instead, and suggest that I try providing more context in my query.\n\nQuery: "' + message + '"\n\nAssistant:'
            result = run_llama(llm, failureprompt, 0.9, max_tokens)
            result_text = result['choices'][0]['text']

            result_text = split_response(result_text, 'Assistant: ', 'Answer: ')
            
            # print(result_text)

        # if there are answers, present them as context to the user
        else:
            result_text = ''

            sources = []

            for answer in answers:
                # split the answer at Answer: and insert into the prompt
                text = split_response(answers[answer]['text'], 'Assistant: ', 'Answer: ')
                result_text += '\n' + text + '\n'
                # add the source to the cited sources list
                sources.append(answers[answer]['source'])

            # print(f'Generated answers: {result_text}')

            # debugging
            # print(answer_prompt)

    return result_text, sources

# answer the user's query
def answer_query(llm, vector_model, message, history, rag, chat_summary, chat_id, context_id, solr_session, vector_session, agent_context, answers_limit, max_tokens, max_prompt_length, score_threshold, max_results_per_search, max_search_results, search_keywords, safe_mode):

    sources = None

    # if there is a history, append the message and result text to it
    history = history or []
    # trim history to last 5 messages
    if len(history) > 5:
        history = history[-5:]
    s = list(sum(history, ()))
    prompt = '\n\n'.join(s)
    # if the prompt is over 9000, truncate it by removing the first n characters until the string is 9000
    while len(prompt) > max_prompt_length:
        prompt = prompt[1:]

    # if the user turned on RAG
    if rag:
        # check the query if they turned on safe mode
        if safe_mode:
            if unsafe_query(llm, message, max_tokens, agent_context):
                prompt += f'\nUser: "{message}"\n\nInstruct the user to kindly not change the subject. You are here to asisst with {agent_context} only.\n\nAssistant: '
                result = run_llama(llm, prompt, 0.5, max_tokens)
                result_text = result['choices'][0]['text']
                result_text = split_response(result_text, 'Assistant: ', 'Answer: ')
                history.append((message, result_text))
                update_db(vector_model, message, result_text.lstrip(), chat_summary.lstrip(), chat_id, solr_session, vector_session)
                return history, chat_summary
        
        # cut the prompt down again by 1/2 to give space for the RAG context
        while len(prompt) > max_prompt_length / 2:
            prompt = prompt[1:]
        # generate a response using the RAG model
        rag_response, sources = rag_query(llm, vector_model, message, solr_session, vector_session, chat_id, context_id, agent_context, answers_limit, max_tokens, score_threshold, max_results_per_search, max_search_results, search_keywords)
        # append the RAG response to the prompt
        prompt += rag_response
    
    if not rag and agent_context and agent_context != '< None >':
        prompt += f'\nReply to the following in the context of {agent_context}.\n'

    # append the user's query to the prompt
    prompt += '\nUser: ' + message + '\n\nAgent: '

    # trim down the prompt once more for good measure
    while len(prompt) > max_prompt_length:
        prompt = prompt[1:]

    # print('Generated prompt: ' + prompt)
    # run the prompt through the LLM
    result = run_llama(llm, prompt, 0.5, max_tokens)
    result_text = result['choices'][0]['text']

    # split the response from the rest of the output that isn't needed
    result_text = split_response(result_text, 'Agent: ', 'Agent: ')

    # add sources to the final response if they were found
    if sources:
        result_text += '\n\nSources:'
        for source in sources:
            result_text += '\n' + source
    
    result_text += '\n'

    # append the result text to the in-memory history
    history.append((message, result_text))

    if chat_summary == 'New Chat':
        chat_summary = summarize_chat(llm, message, result_text, max_tokens)

    # update the databases with the user's query and the chatbot's response
    update_db(vector_model, message, result_text.lstrip(), chat_summary.lstrip(), chat_id, solr_session, vector_session)

    # print(f'Generated response: {result_text}')
    
    return history, chat_summary

# return a dictionary of the conversations in the database
def get_conversation_list(solr_session):
    tables = []
    data = {}

    # get the table names from the conversations keyspace
    solr_query = f"USE system_schema"
    solr_session.execute(solr_query)
    solr_query = f"SELECT table_name FROM tables where keyspace_name = '{CONVO_KEYSPACE}'"
    for row in solr_session.execute(solr_query):
        tables.append(row.table_name)
    
    # get the conversation summary from each table
    for table in tables:
        solr_query = f"SELECT summary FROM {CONVO_KEYSPACE}.{table} LIMIT 1"
        for row in solr_session.execute(solr_query):
            data[table] = row.summary

    return data

# delete a conversation from the database
def delete_conversation(chat_id, solr_session, vector_session):
    solr_session.execute(f"DROP TABLE IF EXISTS {CONVO_KEYSPACE}.{chat_id}")
    vector_session.execute(f"DROP TABLE IF EXISTS {CONVO_KEYSPACE}.{chat_id}")
    print(f'Deleted conversation {chat_id}')

# get the conversation history for a chat_id
def get_conversation(chat_id, solr_session):
    history = []
    data = {}

    # remove the encapsulating $$ from the prompt and response
    for row in solr_session.execute(f"SELECT * FROM {CONVO_KEYSPACE}.{chat_id}"):
        prompt = row.prompt.replace('$$', '')
        response = row.response.replace('$$', '')
        timestamp = row.datetime
        data[timestamp] = (prompt, response)

    # sort the data by timestamp
    for key in sorted(data.keys()):
        history.append(data[key])

    return history

# remove any uploaded books from the database
def unload_books(table_name, solr_session, vector_session):
    solr_session.execute(f"DROP TABLE IF EXISTS {REF_KEYSPACE}.{table_name}")
    vector_session.execute(f"DROP TABLE IF EXISTS {REF_KEYSPACE}.{table_name}")
    print(f'Dropped {table_name} sources')

# load the books into the database
def load_books(context_id, solr_session, vector_session, agent_context, files):
    create_search_agent(solr_session, context_id)
    create_vector_agent(vector_session, context_id)
    insert_books(solr_session, vector_session, context_id, agent_context, files)

# remove any saved conversations from the database
def unload_conversations(solr_session, vector_session):
    solr_session.execute(f"DROP KEYSPACE IF EXISTS {CONVO_KEYSPACE}")
    vector_session.execute(f"DROP KEYSPACE IF EXISTS {CONVO_KEYSPACE}")
    print('Dropped conversations keyspaces')
    create_keyspaces()
    return None, None

# rename a conversation table
def mv_conversation(table_name, new_table, new_summary, solr_session, vector_session):
    # create a new table with the new name
    create_new_conversation(new_table, solr_session, vector_session)
    # copy the data from the old table to the new table
    solr_query = f"SELECT * FROM {CONVO_KEYSPACE}.{table_name}"
    for row in solr_session.execute(solr_query):
        solr_query = f"INSERT INTO {CONVO_KEYSPACE}.{new_table} (id, datetime, summary, prompt, response) VALUES (%s, %s, %s, %s, %s)"
        solr_session.execute(solr_query, (row.id, row.datetime, new_summary, row.prompt, row.response))
    solr_session.execute(f"DROP TABLE IF EXISTS {CONVO_KEYSPACE}.{table_name}")
    # repeat for vector db
    vector_query = f"SELECT * FROM {CONVO_KEYSPACE}.{table_name}"
    for row in vector_session.execute(vector_query):
        vector_query = f"INSERT INTO {CONVO_KEYSPACE}.{new_table} (id, datetime, summary, prompt, prompt_vector) VALUES (%s, %s, %s, %s, %s)"
        vector_session.execute(vector_query, (row.id, row.datetime, new_summary, row.prompt, row.prompt_vector))
    vector_session.execute(f"DROP TABLE IF EXISTS {CONVO_KEYSPACE}.{table_name}")