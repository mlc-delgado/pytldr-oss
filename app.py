from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import torch.cuda as cuda
import gradio as gr
from chatbot import parse_query, answer_query, get_conversation_list, get_conversation, delete_conversation, load_books, unload_books, mv_conversation
from setup_db import  create_keyspaces, connect_database, db_checks, books_checks
from book_data import load_config, write_config
from server_logs import read_logs
import uuid
import re
import os
from huggingface_hub import hf_hub_download

# write config to file
def save_config(max_tokens, max_prompt_length, answers_limit, score_threshold, keywords_limit, max_results_per_search, max_search_results):
    config = load_config()

    config['maxTokens'] = max_tokens
    config['maxPromptLength'] = max_prompt_length
    config['answersLimit'] = answers_limit
    config['scoreThreshold'] = score_threshold
    config['keywordsLimit'] = keywords_limit
    config['maxResultsPerSearch'] = max_results_per_search
    config['maxSearchResults'] = max_search_results
    
    write_config(config)

# select a conversation
def select_conversation(choice):

    chats_list = ['New Chat'] + get_chats()

    if choice == 'New Chat':
        return 'New Chat', 'chat_' + uuid.uuid4().hex[:8], None, None, None, chats_list, False, False

    conversation_history = get_conversation_list(solr_session)

    # get the conversation id from the conversation history
    for id in conversation_history:
        if conversation_history[id] == choice:
            # get the conversation from the database
            history = get_conversation(id, solr_session)
            return conversation_history[id], id, history, history, None, chats_list, False, False

# delete a conversation
def rm_conversation(chat_id):
    delete_conversation(chat_id, solr_session, vector_session)
    conversation_history = get_conversation_list(solr_session)

    # return the first conversation in the list
    try:
        default_choice = conversation_history[list(conversation_history.keys())[0]]
    except IndexError:
        default_choice = 'New Chat'

    return select_conversation(default_choice)

# rename a conversation
def rename_conversation(chat_id, new_name):
    # create a new conversation id and move the conversation to the new id
    new_id = 'chat_' + uuid.uuid4().hex[:8]
    mv_conversation(chat_id, new_id, new_name, solr_session, vector_session)

    return select_conversation(new_name)

# get the list of conversations
def get_chats():
    conversation_history = get_conversation_list(solr_session)
    list = []
    for chat in conversation_history:
        list.append(conversation_history[chat])

    return list

# update the conversation with the new message and reply
def update_conversation(llm, message, state, rag, chat_id, chat_summary, context_id, agent_context, answers_limit, max_tokens, max_prompt_length, score_threshold, max_results_per_search, max_search_results, search_terms, safe_mode):
    history, summary = answer_query(llm, transformer, message, state, rag, chat_summary, chat_id, context_id, solr_session, vector_session, agent_context, answers_limit, max_tokens, max_prompt_length, score_threshold, max_results_per_search, max_search_results, search_terms, safe_mode)

    return message, history, history, summary, ['New Chat'] + get_chats(), []

def table_name(context):
    id = re.sub(r'\W+', '', context.lower())
    # if the context starts with a number, add a prefix
    if id[0].isdigit():
        id = 'context_' + id

    return id

# drop any loaded books for a given context
def drop_books(context):

    id = table_name(context)

    unload_books(id, solr_session, vector_session)

    return False

# load the books from the directory
def init_books(context, files):

    drop_books(context)

    data = load_config()['agentContexts']
    for id in data:
        if data[id]['displayName'] == context:
            load_books(id, solr_session, vector_session, context, files)
            return files, books_checks(solr_session, id)

# return the opposite state
def toggle_state(state):
    return not state

# list the llms in the directory
def list_llms(directory):
    llms = []
    for file in os.listdir(directory):
        if file.endswith('.gguf') and 'ggml' not in file:
            llms.append(file)

    return llms

# load a LLM from file
def load_llm(llm_file, llm, chatbot):
    
    # remove the existing llm from memory
    del llm

    if llm_file == []:
        return [], None, chatbot

    if '7b' in llm_file.lower():
        gpu_layers = 35
    elif '8b' in llm_file.lower():
        gpu_layers = 36
    elif '13b' in llm_file.lower():
        gpu_layers = 43
    elif '30b' in llm_file.lower():
        gpu_layers = 61

    llm_path = load_config()['llmPath']
    llm_filepath = llm_path + llm_file
    llama_llm = Llama(
        model_path=llm_filepath,
        n_gpu_layers=gpu_layers,
        main_gpu=0, 
        n_cpu_threads=4, 
        verbose=True, 
        n_ctx=load_config()['maxTokens']
        )
    return llm_file, llama_llm, chatbot

# load a transformer model
def load_transformer(model):
    transformer = SentenceTransformer(
        model,
        device = 'cuda' if cuda.is_available() else 'cpu'
    )

    return transformer

# load an agent context to the chatbot
def load_context(context):
    # return the default state if no context is selected
    if context == '< None >':
        return context, None, False, []

    data = load_config()['agentContexts']
    for id in data:
        if data[id]['displayName'] == context:
            if data[id]['omitKeywords']:
                return context, id, books_checks(solr_session, id), data[id]['omitKeywords']
            else:
                return context, id, books_checks(solr_session, id), []

# load display names for agent contexts
def load_context_display_names():
    displayNames = []
    data = load_config()['agentContexts']
    if data:
        for context in data:
            displayNames.append(data[context]['displayName'])

    return displayNames

# load the agent context settings
def load_context_settings(context):
    if context == '< New >':
        return {}, False, False, context
    data = load_config()['agentContexts']
    for id in data:
        if data[id]['displayName'] == context:
            return data[id], books_checks(solr_session, id), False, context

# save the agent context settings
def save_context_settings(selection, display_name, omit_keywords):
    config = load_config()

    # check if the display name is empty and don't save if it is
    if display_name == '' or display_name == None:
        return load_context_settings('< New >'), load_context_display_names()
    
    # split the omit keywords into a list if provided
    if omit_keywords == '':
        omit_keywords = None
    else:
        omit_keywords = omit_keywords.split(', ')
    
    id = table_name(display_name)

    # write a new context if the selection is '< New >'
    if selection == '< New >':
        config['agentContexts'][id] = {
            'displayName': display_name,
            'omitKeywords': omit_keywords
        }
        write_config(config)
    # overwrite the existing context
    else:
        config['agentContexts'][id] = {
            'displayName': display_name,
            'omitKeywords': omit_keywords
        }
        
        write_config(config)

    agent_settings, books_loaded, delete_agent, selected_context = load_context_settings(display_name)

    return agent_settings, books_loaded, delete_agent, selected_context, load_context_display_names()

# return the selected state
def set_state(state):
    return state

# delete an agent context
def delete_context(context):
    config = load_config()
    data = config['agentContexts']
    for id in data:
        if data[id]['displayName'] == context:
            del config['agentContexts'][id]
            write_config(config)
            # remove any books associated with the context
            unload_books(id, solr_session, vector_session)
            break    

    agent_settings, books_loaded, delete_agent, selected_context = load_context_settings('< New >')

    return agent_settings, books_loaded, delete_agent, selected_context, load_context_display_names()

# download a LLM from Hugging Face
def download_llm(repo_id, filename):
    print('Downloading', filename, 'from', repo_id)
    path = load_config()['llmPath']

    try:
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=path,
        )
        print('Finished downloading', downloaded_file)
    except Exception as e:
        print('Error downloading model:', e)
        print('Ensure the repo ID and filename are correct')
        return repo_id, filename, list_llms(path)

    return None, None, list_llms(path)

with gr.Blocks() as demo:

    # set initial states
    books_loaded = gr.State(value=False)
    rag_checkbox_enabled = gr.State(value=False)
    rag = gr.State(value=False)
    agent_context = gr.State(value='< None >')
    state = gr.State(value=[])
    edit_summary = gr.State(value=False)
    delete_convo = gr.State(value=False)
    llm = gr.State(value=None)
    selected_llm = gr.State(value=None)
    agent_settings = gr.State(value={})
    delete_agent = gr.State(value=False)
    default_llms = gr.State(value=list_llms(load_config()['llmPath']))
    context_display_names = gr.State(value=load_context_display_names())
    selected_context = gr.State(value='< New >')
    omit_keywords = gr.State(value=[])
    enable_advanced_rag = gr.State(value=False)
    search_terms = gr.State(value=[])
    context_id = gr.State(value=None)
    chat_summary = gr.State(value='New Chat')
    chat_id = gr.State(value='chat_' + uuid.uuid4().hex[:8])
    connected = gr.State(value=False)
    safe_mode = gr.State(value=False)

    # load the config values from file
    default_config = load_config()
    answers_limit = gr.State(value=default_config['answersLimit'])
    keywords_limit = gr.State(value=default_config['keywordsLimit'])
    max_tokens = gr.State(value=default_config['maxTokens'])
    max_prompt_length = gr.State(value=default_config['maxPromptLength'])
    score_threshold = gr.State(value=default_config['scoreThreshold'])
    max_results_per_search = gr.State(value=default_config['maxResultsPerSearch'])
    max_search_results = gr.State(value=default_config['maxSearchResults'])
    solr_db_host = gr.State(value=default_config['solrHost'])
    solr_db_port = gr.State(value=default_config['solrPort'])
    vector_db_host = gr.State(value=default_config['vectorHost'])
    vector_db_port = gr.State(value=default_config['vectorPort'])

    # load the database sessions and transformer
    # these cannot be stored as Gradio states
    global transformer, solr_session, vector_session
    transformer = load_transformer('all-MiniLM-L6-v2')
    solr_session = connect_database(solr_db_host.value, solr_db_port.value)
    vector_session = connect_database(vector_db_host.value, vector_db_port.value)

    # set up initial values so that conversations can be saved
    if solr_session and vector_session:
        connected.value = True
        ref_keyspace_exists, convo_keyspace_exists = db_checks(solr_session)

        # create the keyspaces if they do not exist
        if not ref_keyspace_exists or not convo_keyspace_exists:
            create_keyspaces(solr_session, vector_session)

        # append historical chats to the list of chats
        chats_list = gr.State(value=[chat_summary.value] + get_chats())
    else:
        chats_list = gr.State(value=[chat_summary.value])

    with gr.Tab('Chatbot'):
        
        with gr.Row():
            with gr.Column(scale=1, min_width=400):
                gr.Markdown(f'## History')
                # update the chat list as new conversations are added, edited, or removed
                @gr.render(inputs=chats_list)
                def show_chat_list(chats):
                    for chat in chats:
                        with gr.Row():
                            # add button to select a conversation
                            select_conversation_button = gr.Button(chat, size='sm')
                            select_conversation_button.click(select_conversation, inputs=select_conversation_button, outputs=[chat_summary, chat_id, state, chatbot, message, chats_list, edit_summary]) 

            # Add the chatbot interface
            with gr.Column(scale=3):
                # live update the chatbot with the summary of the conversation
                @gr.render(inputs=[chat_summary, edit_summary, delete_convo])
                def show_chat_summary(summary, edit, delete):
                    gr.Markdown(f'## {summary}')
                    if summary != 'New Chat':
                        # add rename and edit buttons for hisorical chats
                        if edit:
                            new_summary = gr.Textbox(label='New conversation name', scale=2)
                            with gr.Row():
                                gr.Button('Submit').click(rename_conversation, inputs=[chat_id, new_summary], outputs=[chat_summary, chat_id, state, chatbot, message, chats_list, edit_summary, delete_convo])
                                gr.Button('Cancel', icon='images/delete.png').click(toggle_state, inputs=edit_summary, outputs=edit_summary)
                        elif delete:
                            with gr.Row():
                                gr.Markdown(f'### Are you sure you want to delete "{summary}"?')
                            with gr.Row():
                                gr.Button('Yes').click(rm_conversation, inputs=chat_id, outputs=[chat_summary, chat_id, state, chatbot, message, chats_list, edit_summary, delete_convo])
                                gr.Button('No').click(toggle_state, inputs=delete_convo, outputs=delete_convo)
                        else:
                            with gr.Row():
                                gr.Button('Rename Chat', icon='images/edit.png').click(toggle_state, inputs=edit_summary, outputs=edit_summary)
                                gr.Button('Delete Chat',icon='images/delete.png').click(toggle_state, inputs=delete_convo, outputs=delete_convo)
    
                # add selectors for the LLM and agent context
                @gr.render(inputs=selected_llm)
                def show_llm(selected):
                    if selected:
                        llm_select = gr.Dropdown(value=selected, label='LLM', choices=list_llms(load_config()['llmPath']), scale=2)
                    else:
                        llm_select = gr.Dropdown(label='LLM', choices=list_llms(load_config()['llmPath']), scale=2)
                    llm_select.change(load_llm, inputs=[llm_select, llm, chatbot], outputs=[selected_llm, llm, chatbot])
                
                @gr.render(inputs=[context_display_names])
                def show_contexts(display_names):
                    context_select = gr.Dropdown(value='< None >',label='Agent Context', choices=['< None >'] + display_names, scale=2)
                    context_select.change(load_context, inputs=context_select, outputs=[agent_context, context_id, rag_checkbox_enabled, omit_keywords])

                # display the chatbot and message input so users can view conversations without LLM loaded
                chatbot = gr.Chatbot()
                message = gr.Textbox(placeholder='Type your query here...', container=False)

                # add a markdown to display the search terms that were returned
                @gr.render(inputs=[search_terms, rag])
                def show_search_terms(terms, rag):
                    if terms != [] and rag:
                        gr.Markdown(f'### Searching {", ".join(terms)}')

                # add a checkbox to enable RAG queries
                @gr.render(inputs=[rag_checkbox_enabled])
                def show_books_loaded(loaded):
                    if loaded:
                        search_enable = gr.Checkbox(label='Search my data')
                        search_enable.input(set_state, inputs=search_enable, outputs=rag)

                # add additional chatbot inputs if a LLM is selected
                @gr.render(inputs=[selected_llm, connected, rag])
                def show_chatbot(selected, db, sources):
                    if selected and db:
                        submit=gr.Button('Submit')
                        # parse the query if RAG is enabled
                        if sources: 
                            submit.click(parse_query, inputs=[llm, message, agent_context, omit_keywords, keywords_limit, max_tokens], outputs=[search_terms]).then(update_conversation, inputs=[llm, message, state, rag, chat_id, chat_summary, context_id, agent_context, answers_limit, max_tokens, max_prompt_length, score_threshold, max_results_per_search, max_search_results, search_terms, safe_mode], outputs=[message, chatbot, state, chat_summary, chats_list, search_terms])
                        # just update the conversation if RAG is not enabled
                        else:
                            submit.click(update_conversation, inputs=[llm, message, state, rag, chat_id, chat_summary, context_id, agent_context, answers_limit, max_tokens, max_prompt_length, score_threshold, max_results_per_search, max_search_results, search_terms, safe_mode], outputs=[message, chatbot, state, chat_summary, chats_list, search_terms])
                    elif selected and not db:
                        gr.Markdown('### No database connection')
                        gr.Button('Submit', interactive=False)
                    elif db and not selected:
                        gr.Markdown('### No model loaded')
                        gr.Button('Submit', interactive=False)
                    gr.ClearButton([message])                   

    with gr.Tab('Agent Settings'):

        with gr.Row():
            with gr.Column():
                gr.Markdown(f'## Agent Settings')
                # display the agent context settings
                @gr.render(inputs=[context_display_names, selected_context])
                def show_contexts(display_names, selected):
                    context_select2 = gr.Dropdown(value=selected, label='Agent Context', choices=['< New >'] + display_names, scale=2)
                    context_select2.change(load_context_settings, inputs=context_select2, outputs=[agent_settings, books_loaded, delete_agent, selected_context])
                
                @gr.render(inputs=[agent_settings, delete_agent])
                def show_agent_settings(settings, delete):
                    if settings != {}:
                        display_name = gr.Textbox(label='Name', type='text', value=settings['displayName'], scale=2, interactive=False)
                        if settings['omitKeywords']:
                            keywords = gr.Textbox(label='Omit Search Keywords', type='text', value=', '.join(settings['omitKeywords']), scale=2)
                        else:
                            keywords = gr.Textbox(label='Omit Search Keywords', type='text', scale=2)
                        if delete:
                            gr.Markdown(f'### Are you sure you want to delete "{settings["displayName"]}"? This will also remove any uploaded sources.')
                            with gr.Row():
                                gr.Button('Yes').click(delete_context, inputs=selected_context, outputs=[agent_settings, books_loaded, delete_agent, selected_context, context_display_names])
                                gr.Button('No').click(toggle_state, inputs=delete_agent, outputs=delete_agent)
                        else:        
                            gr.Button('Save Agent Context').click(save_context_settings, inputs=[selected_context, display_name, keywords], outputs=[agent_settings, books_loaded, delete_agent, selected_context, context_display_names])
                            gr.Button('Delete Context').click(toggle_state, inputs=delete_agent, outputs=delete_agent)
                    else:
                        display_name=gr.Textbox(label='Name', type='text', scale=2)
                        keywords=gr.Textbox(label='Omit Search Keywords', type='text', scale=2)
                        gr.Button('Create New Context').click(save_context_settings, inputs=[selected_context, display_name, keywords], outputs=[agent_settings, books_loaded, delete_agent, selected_context, context_display_names])            

            with gr.Column():
                gr.Markdown('## Data Sources')
                with gr.Row():
                    with gr.Column():
                        # provide a selector for files to upload
                        upload_files = gr.File(
                            label='Select files to upload',
                            file_count='multiple'
                        )
                        # Add buttons to load and remove sources
                        @gr.render(inputs=[upload_files, selected_context, books_loaded])
                        def enable_button(upload, selection, loaded):
                            if selection != '< New >':
                                if loaded:
                                    gr.Markdown(f'### Sources loaded')
                                    if upload:
                                        book_load = gr.Button('Reload Sources', interactive=True)
                                        book_load.click(init_books, inputs=[selected_context, upload_files], outputs=[upload_files, books_loaded])
                                    else:
                                        book_load = gr.Button('Reload Sources', interactive=False)
                                    book_unload = gr.Button('Remove Sources')
                                    book_unload.click(drop_books, inputs=selected_context, outputs=books_loaded)
                                else:
                                    gr.Markdown(f'### No sources loaded')
                                    if upload:
                                        book_load = gr.Button('Load Sources', interactive=True)
                                        book_load.click(init_books, inputs=[selected_context, upload_files], outputs=[upload_files, books_loaded])
                                    else:
                                        book_load = gr.Button('Load Sources', interactive=False)
                                    book_unload = gr.Button('Remove Sources', interactive=False)
                                    
        # display background activity in a console
        with gr.Row():
            logs = gr.Code(label='Console Output')
            demo.load(read_logs, None, logs, every=1)

    with gr.Tab('RAG Settings'):

        with gr.Row():
            with gr.Column():
                @gr.render(inputs=default_llms)
                def show_llm_settings(llms):
                    # Provide option to download LLM if default does not exist
                    gr.Markdown('## Download Model')
                    if 'llama-2-13b-chat.Q5_K_M.gguf' not in llms:
                        gr.Markdown('### LLaMA 2-13B model not found')
                        repo_id = gr.Textbox(label='Repo ID', type='text', value='TheBloke/Llama-2-13B-chat-GGUF', info='Hugging Face repo ID')
                        filename = gr.Textbox(label='Filename', type='text', value='llama-2-13b-chat.Q5_K_M.gguf', info='Filename of the model')
                        
                    else:
                        # Display available LLMs if default already exists
                        gr.Dropdown(label='Available LLMs', choices=llms, scale=2)
                        repo_id = gr.Textbox(label='Repo ID', type='text', placeholder='Enter huggingface repo id, e.g. TheBloke/Llama-2-13B-chat-GGUF', info='Hugging Face repo ID')
                        filename = gr.Textbox(label='Filename', type='text', placeholder='Enter filename of the model, e.g. llama-2-13b-chat.Q5_K_M.gguf', info='Filename of the model')
                    gr.Button('Download Model').click(download_llm, inputs=[repo_id, filename], outputs=[repo_id, filename, default_llms])
                    
                    
            with gr.Column():
                gr.Markdown(f'## RAG Settings')
                # Basic RAG settings
                @gr.render(inputs=[answers_limit, keywords_limit, enable_advanced_rag, safe_mode])
                def show_rag_settings(answers, keywords, enable, safe):
                    answers_limit_value = gr.Slider(label='Answers Limit', value=answers, info='Maximum number of answers to use as RAG context', minimum=1, maximum=20, step=1)
                    answers_limit_value.change(set_state, inputs=answers_limit_value, outputs=answers_limit)
                    keywords_limit_value = gr.Slider(label='Keywords Limit', value=keywords, info='Maximum number of keywords to use in the Solr search', minimum=1, maximum=20, step=1)
                    keywords_limit_value.change(set_state, inputs=keywords_limit_value, outputs=keywords_limit)
                    safe_mode_value = gr.Checkbox(value=safe, label='Safe Mode', info='Enable Safe Mode to prevent users from going outside of the agent context with RAG')
                    safe_mode_value.change(set_state, inputs=safe_mode_value, outputs=safe_mode)

                    # Advanced RAG settings
                    advanced_rag = gr.Checkbox(value=enable, label='Show Advanced RAG Settings')
                    advanced_rag.input(set_state, inputs=advanced_rag, outputs=enable_advanced_rag)
                    if enable:
                        gr.Markdown('### WARNING: Editing these values may negatively impact performance or result in in the app becoming unusable. Do not change these unless you are sure of what you are doing.')
                        max_tokens_value = gr.Slider(label='Max Tokens', value=default_config['maxTokens'], info='Maximum number of tokens for LLM', minimum=1024, maximum=8192, step=512)
                        max_tokens_value.change(set_state, inputs=max_tokens_value, outputs=max_tokens)
                        max_prompt_length_value = gr.Slider(label='Max Prompt Length', value=default_config['maxPromptLength'], info='Maximum number of characters in prompts', minimum=1000, maximum=16000, step=500)
                        max_prompt_length_value.change(set_state, inputs=max_prompt_length_value, outputs=max_prompt_length)
                        score_threshold_value = gr.Slider(label='Score Threshold', value=default_config['scoreThreshold'], info='Minimum cosine similiarity score for answers', minimum=0.0, maximum=1.0, step=0.01)
                        score_threshold_value.change(set_state, inputs=score_threshold_value, outputs=score_threshold)
                        max_results_per_search_value = gr.Slider(label='Max Results Per Search', value=default_config['maxResultsPerSearch'], info='Maximum number of results to return per Solr search', minimum=1, maximum=20, step=1)
                        max_results_per_search_value.change(set_state, inputs=max_results_per_search_value, outputs=max_results_per_search)
                        max_search_results_value = gr.Slider(label='Max Search Results', value=default_config['maxSearchResults'], info='Maximum number of Solr search results in total', minimum=1, maximum=100, step=5)
                        max_search_results_value.change(set_state, inputs=max_search_results_value, outputs=max_search_results)

                save_settings = gr.Button('Save Settings')
                save_settings.click(save_config, inputs=[max_tokens, max_prompt_length, answers_limit, score_threshold, keywords_limit, max_results_per_search, max_search_results]) 

# run chatbot server
if __name__ == '__main__':
    demo.queue().launch(debug=True)