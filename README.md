# PyTLDR
![cover](./images/pytldr_cover_small.png "cover")

Welcome to PyTLDR, a Python project that pushes the boundaries of chatbot assistants by combining NoSQL databases, language models, and prompt engineering. In this README file, I'll provide an in-depth overview of the system's architecture and functionality, as well as installation and usage instructions.

If you have ever wanted to deploy a private version of ChatGPT that can index and read your ebooks and text data, then this is the app for you. Best of all, it runs locally on your hardware, and runs on a wide range of graphics processing units (GPUs), no third party subscriptions or APIs necessary.

## Purpose
Many professionals are interested in integrating Large Language Models (LLMs) into their tooling and daily workflow, but are concerned about privacy and cost. Additionally, LLMs are prone to hallucniation, and it is extremely costly to train and fine tune your own models for your particular use case. This application alleviates those concerns by allowing users to run inference locally on hardware they own, and integrates with Cassandra databases to provide Retrieval Augmented Generation (RAG), allowing users to augment foundation models such as Llama 2 with domain-specific knowledge, without investing in training or fine tuning.

By instructing the LLM to respond to queries based on data you own, the possibilities for education, systems automation, and accelerating development are endless. Sit down and have a conversation with your data, and never fear the mountains of text you need to consume in order to go about your day. If there is no time to read or consume it, now you can simply ask your virtual assistant to do the dirty work for you.

![Main UI](./images/full_window.png "Main UI")

## Architecture Overview

PyTLDR consists of four primary components:

1. **Cassandra Databases**: Utilizes Cassandra's distributed architecture to store ebook metadata and page-level embeddings, enabling efficient querying and retrieval.
2. **Large Language Model**: Leverages Llama 2's powerful language understanding capabilities for summarization, paraphrasing, and query-answer matching.
3. **Sentence Transformer**: Utilizes a state-of-the-art sentence transformer model for generating embeddings that capture semantic relationships between sentences.
4. **Prompt Engineering**: Develops custom prompts to guide the LLM's processing of user queries, ebook content, and search results, allowing for fine-grained control over the generation process.

## Functionality

Here's how retrieval augmented generation (RAG) in PyTLDR works:

1. **User Query Processing**: Users enter a query, which is processed into a set of search keywords and a paraphrasing of the request.
2. **Solr DB Queries**: Executes Solr queries against the Cassandra database using the search keywords to retrieve relevant sources, leveraging the power of the Lucene engine.
3. **Contextual answering**: For each result, compares the source and the user's original query, and discards irrelevant results.
4. **Vector Similarity Ranking**: Ranks remaining results by similarity to the user's prompt, utilizing vector embeddings for efficient similarity measurement.
5. **Context Generation**: Analyzes, extracts, and paraphrases the relevant data from each source to be presented as prompt context.
6. **Response Generation**: Answers the user's query using the provided sources, adjusting outputs to user preferences.

## Key Features

PyTLDR offers several key advantages over traditional chatbots and information retrieval systems:

* **Domain-specific knowledge**: Gleans domain-specific knowledge representations from text and ebook content without any training or fine-tuning.
* **Control over context**: Allows users to control how much context is provided to the LLM, allowing for more precise answers.
* **Rolling context window**: The newest conversation replies are automatically stored in context for the next prompt, ensuring that the conversation can continue for as long as you have free storage.
* **Query-answer matching**: Compares each query and answer with the LLM to determine whether a result is actually relevant before providing it as context.
* **Compatibility**: Requires minimal Python dependencies and runs on any supported LLMs and GPUs for llama-cpp-python, including major brands such as Nvidia, AMD, Intel, and Apple Silicon.

## Flowchart

![Flowchart](./images/pytldr_flowchart.png "Flowchart")

# Installation
Use the following instructions to set up the environment and dependencies for PyTLDR. You will need some familiarity with Linux command line, Python, and Docker in order to proceed, so if you are unfamiliar please refer to the official documentation for these tools.

## Prerequisites
* An Ubuntu 22.04 host system, or Ubuntu 22.04 virtual machine with GPU access, such as Windows Subsystem for Linux (WSL)
* [Git](https://github.com/git-guides/install-git#install-git-on-linux)
* [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) and [docker compose plugin](https://docs.docker.com/compose/install/linux/) for Linux
* At least 16 GiB RAM
* At least 60 GiB free space for storing LLMs and Docker images, plus additional storage for databases
* A GPU with at least 12 GiB of VRAM and supporting CUDA 12.4 or ROCm 6.1 (Tested with Nvidia Ampere and AMD RDNA3 series GPUs)
* A stable internet connection for downloading dependencies and LLMs

## Setting up environment
1. On the host operating system, install a compatible AMD ROCm or Nvidia CUDA graphics driver. The instructions for this will differ depending on your hardware vendor and host operating system. Refer to the manufacturer instructions for installing the drivers.
2. On the Ubuntu 22.04 system (if different from host operating system), install a compatible version of Docker Engine and docker compose plugin as in the links above. Note that Nvidia GPUs may require a specific distribution of Docker Engine from Nvidia, so please refer to their instructions for more information.
3. Download the files from the repository, or open a terminal in Ubuntu and clone the repository with git.
```
git clone https://github.com/mlc-delgado/pytldr-oss.git
```
4. With the terminal open, then navigate to the `/pytldr-oss` folder for the next steps.

## Configure local volumes
This app works best with persistent data storage for LLMs and sentence transformer models. Since the app runs within Docker, you will need to instruct Docker on where to locate the files that will persist outside of Docker.
1. Open the docker compose YAML file that corresponds to your hardware in your preferred text editor or IDE. Use the `docker-compose-rocm.yml` for AMD, and if using Nvidia open the `docker-compose-cuda.yml`.
2. Decide where you would like to store your Cassandra data, LLM files, and sentence transformer files. You will need to define these local directories under the volumes for each service. For ease of use, you can store the Cassandra data under directories in `pytldr-oss/dse`, using the included creation script at its default location. Use the format `< local directory >:< container directory >` when setting up volumes. For example:

```
volumes:
    - /home/< your username >/code/pytldr-oss/6.8.37:/var/lib/cassandra
...
volumes:
    - /home/< your username >/code/pytldr-oss/7.0.0-alpha.4:/var/lib/cassandra
```
3. Repeat this for both databases and the pytldr server. Map the `/pytldr` directory to where you cloned the Python code, and the `/usr/share/models` to a downloads directory of your choosing. The `/root/.cache/huggingface` can be another directory of your choosing, or you can map it to your existing Hugging Face cache if you've downloaded sentence transformer models in the past. If you prefer to use a different path for storing the LLMs in the container, make sure to update the `llmPath` in the `config.yaml` to reflect the new download path for LLMs. For example:
```
volumes:
    - /home/< your username >/code/pytldr-oss:/pytldr
    - /home/< your username >/code/llama.cpp/models:/usr/share/models
    - /home/< your username >/.cache/huggingface:/root/.cache/huggingface
```

## Permissions and storage for Cassandra
***IMPORTANT:*** These steps are only required when setting up a PyTLDR installation for the first time, in order to prevent errors when loading DataStax Enterprise containers. The default images for DataStax Enterprise expect there to be existing directories and permissions within `/var/lib/cassandra` for a user named `dse`. When the Cassandra data volume is mapped to the host machine, the containers do not automatically create the `dse` user and data directories. To work around this, these steps will create the user and directories required in order for the containers to start. You will need `sudo` privileges or execute as root to proceed with these steps.
1. Navigate to the `pytldr-oss/dse` directory.
```
cd dse
```
2. Execute the following to create the dse user and required directories. 
```
sudo python3 create_folders.py
```
3. This should create two directories, `6.8.37` and `7.0.0-alpha.4`, and several subdirectories, in the current working directory. If you run into any errors during the creation of the user and directories and want to start fresh, you will need to use `sudo` again to clean these up. The following commands will remove the  created user and the data directories:
```
sudo userdel dse
sudo rm -r 6.8.37
sudo rm -r 7.0.0-alpha.4
```

## Build and launch Docker image
1. Select the dockerfile to build that matches your hardware. Again if using AMD GPUs, build from the `pytldr-rocm.dockerfile`, and if using Nvidia use the `pytldr-cuda.dockerfile`.
```
docker build -t pytldr -f pytldr-< platform >.dockerfile
```
2. After building the image, launch with the updated docker-compose YAML that matches your hardware.
```
docker compose -f docker-compose-< platform >.yml up -d
```
3. The web server will wait for the databases to become available before it starts up. Once it starts, you can watch the logs and monitor the status of the server with the following:
```
docker logs --follow pytldr-server-1
```
4. Wait for the following message to appear in the logs to confirm the server is up:
```
Running on local URL:  http://0.0.0.0:7860
```
5. To start using the app, open a web browser tab and navigate to the URL http://0.0.0.0:7860, or http://127.0.0.1:7860 if using Windows Subsystem for Linux (WSL).

## Stopping the server
1. Enter `CTRL+C` to stop watching the docker logs, or from another terminal execute the following command to gracefully shut down the server and databases:
```
docker compose -f docker-compose-< platform >.yml down
```
2. ***CAUTION***: Wait for the containers to completely stop before shutting down or restarting your host system. The Cassandra databases are highly sensitive to interruptions, and you may risk data corruption if they are not shut down properly. For large and production settings it is best to set up a dedicated Cassandra cluster instead.

# Usage
Your data and conversations will persist on the Cassandra databases between application restarts. On first startup you will simply be presented with a menu to select a LLM and context, and from there you can start a conversation. If you have previous saved conversations then they will be stored in the **History**. Click on the name of a conversation to switch to another conversation, or click on **New Chat** to start a new one.

If you have not already downloaded a compatible LLM, the LLM list will be empty. To download the default model, click on the **RAG Settings** tab and see the **Downloading models** section below before starting.

![Default Page](./images/home_page.png "Default Page")

## Prompting
1. Select a LLM from the drop-down list and wait for it to load. Depending on your hard drive speed and GPU hardware, it may take a few minutes for the model to load. Please be patient, and you will not be able to submit prompts until the model is fully loaded.
2. Select an agent context from the drop-down list, or leave as `< None >`.
3. (Optional) If the agent context has sources loaded, enable `Search my data` to enable the RAG workflow.
4. Enter your prompt and click on `Submit`, then wait for the response to be generated. If RAG is enabled, a list of the generated search keywords will temporarily appear while the search is in progress.
5. After receiving a response, you can click on `Clear` to immediately clear the prompt input box.
6. You can continue prompting to the current conversation, or switch to another on the left pane. Conversation history is stored indefinitely on the Cassandra databases, and there are no hard imposed limits on length or number of conversations. Be sure to keep an eye on disk usage and application performance as your data grows, and remove any unneeded conversations if possible.

![Executing RAG](./images/search_keywords.png "Executing RAG")

## Managing agent contexts
Agent contexts are short, user-defined phrases to describe a subject or a line of thinking that the chatbot should apply when searching and answering queries. These short phrases could be the name of a topic, such as `vegan cooking`, or the name of a product, such as `Ubuntu Linux`. This phrase gets inserted into system prompts to guide the LLM, literally as in `Answer this query in the context of < agent context >.` Defining a new context is as simple as completing the sentence with a name or description of your desired agent context.

`Omit Search Keywords` is an optional list of keywords that you can use to clean up your RAG queries. Some agent contexts, like the rules for Warhammer 40K, have a tendency to cause the LLM to generate a lot of extra search keywords that aren't necessary in the given context, which can skew search results. For those contexts, you can list the keywords you want to omit from searches. Removing keywords such as "Warhammer" and "40K" will greatly improve the accuracy of the searches, since those are already implied by the sources.

Agent contexts are saved to the local `config.yaml` file. Exercise caution when creating and removing context. Once created, an agent context cannot be renamed, and removing an agent context will also remove any sources associated with that context.

![Agent Settings](./images/agent_settings.png "Agent Settings")

## Using agent context

Agent contexts are optional for a conversation. If you leave the agent context as `< None >`, then you are freely chatting with the LLM with no system prompt and no RAG, so when you prompt it will behave the same as the default Llama 2 model. However, once you select an agent context the chatbot behaves completely differently.

Agent contexts can operate in three different modes, **system prompts**, **system prompts with RAG**, and **Safe Mode**.

### System prompts (no RAG)
If you have no books or sources loaded to your selected agent context, or if you have sources but leave the `Search my data` box unchecked, then selecting an agent context other than `< None >` will apply a system prompt that instructs the LLM to respond in the chosen context. It is not searching your sources, but it will attempt to follow the prompt and the current conversation history to control the context of the responses.

### System prompts with RAG
If you have sources loaded to the selected agent context and have checked the box for `Search my data`, then the full RAG workflow activates. When RAG is enabled the system prompts inform the LLM to present only what is found in the source data. Depending on the amount of context found, sometimes the conversation history is ignored in favor of fitting as much context as possible for answers.

Exceeding the maximum number of tokens that the LLM can process with a single prompt can cause the GPU to run out of memory, so the prompt length is given a hard limit and the app truncates everything else. Given this, it's best to enable RAG only when you want to locate information in your sources, and turn it off when not needed. Once you've received a response that contains the data you need, you can switch off `Search my data`, which alllows for more space in the context for the conversation history. From there you can prompt the chatbot again to transform the previous reply into the format you want.

To help lessen the amount of processing of the RAG prompts, the chatbot will attempt to suggest previous answers if a user repeats the same queries in a conversation. This is currently experimental, and will not activate if you rephrase the previous query rather than repeat it.

### Safe Mode
This is the most restrictive mode for RAG, and is disabled by default. **Safe Mode** applies an additional prompt engineering check at the beginning of the RAG workflow to ensure that the answers stay within the selected agent context. Enable with the checkbox labeled `Safe Mode` in the `RAG Settings` menu. You must enable both `Search my data` and `Safe Mode` for the feature to activate.

Under **Safe Mode**, when users issue commands such as, ***"Ignore all previous instructions"***, the chatbot will interpret that as an attempt to change the subject, and will disregard the instructions and try to redirect back to the agent context. This treats all queries outside of the agent context as potentially unsafe, and will prevent the chatbot from answering outside the selected context, rather than responding according to its own definition of an unsafe request.

![Safe Mode Active](./images/unsafe_request.png "Safe Mode Active")

## Managing data
Conversations and data sources are stored in Casssandra databases with DataStax Enterprise containers. This presents a few benefits and a few drawbacks.

On one hand, this allows the application to benefit from the features and performance enhancements provided by DataStax Enterprise. Solr and vector indexing and searching are easy to implement, with little to no configuration required out of the box. Additionally, tools and support are available for migrating to larger Cassandra clusters and enterprise management tools. On the other hand, this adds some complexity to the database infrastructure, and makes it more difficult to start and stop the application from a user's local machine.

Alternate databases are being evaluated and may make it into a future release. For now, be sure to give yourself enough time to gracefully start and stop the databases and application before shutting down the host operating system.

## Loading agent context sources
Source files are linked to the selected agent context when they are uploaded. This ensures that when you perform searches with RAG you can keep conflicting sources and contexts separate from one another.

Currently the following file extensions are supported:
* PDF
* HTML
* EPUB
* TXT
* DOCX
* DOC
* MD

### Additional considerations for sources
* You can only search the sources for a single context at a time. Searches across multiple contexts are not supported.
* You must create or select an agent context before you can upload sources. If you decide to change your sources, you will need to reload all of the sources for that agent context, old and new. Future improvements are planned to allow easier management of sources.
* Book metadata is automatically extracted from EPUB files. For all other file types, only the title and author information are extracted from the file name. You can specify the author in the file name by using the format `< book title > Author < author >.< extension >`, such as `The Old Man and the Sea Author Ernest Hemingway.pdf`.

## Downloading models
The application was developed around Llama 2 at 13 billion parameters. If the default model is not found in the `llmPath` specified in the `config.yaml`, then the application will provide the option to download it for you from Hugging Face.

Alternatively, you can supply your own llama.cpp-compatible GGUF model file, but this can have unpredictable effects on the prompts, as each model performs slightly differently across various prompts. I tested with Mistral and got similar results, but not quite as accurate as Llama 2, and tested again with Llama 3 and saw it become completely unusable. For best results I recommend sticking to the default Llama 2, pending future updates.

## Performance
Without RAG, the app performs the same as other chatbots when prompting Llama 2. Enabling RAG adds to the number of prompts that are executed in the background before a final answer is provided to the user, so you should expect to see a noticeable increase in processing time while RAG is active. For capable hardware, results can usually be generated within 30-90 seconds.

The quality and accuracy of the outputs depends entirely on your data sources and what is being asked. You will need to experiment with search keywords, the number of sources, and various types of prompts in order to determine what works best for your use case.

![RAG Settings](./images/rag_settings.png "RAG Settings")

## Saving Settings
Making changes in the `RAG Settings` menu will change them for the current session only. If you want the settings to persist between application restarts, then click on `Save Settings` to save the changes to the `config.yaml`.

# FAQ

* **The system prompts appear to be really inconsistent in the source code. Why are they configured in this way?**
    * The formats of the system prompts were chosen through rigorous testing. I observed multiple instances where relocating and rephrasing the prompt engineering instructions in various ways throughout the body of the prompt tends to result in wildly different outputs, with no apparent rhyme or reason. Conventional wisdom and existing guides for prompt engineering were eschewed in favor of what works specifically for the version of Llama 2 13B that I used. I cannot guarantee that other models will perform the same.
* **Can I use this to summarize an entire book?**
    * Your mileage may vary if you attempt to ask for a summary. I did not write any specific application logic to handle summarizing entire volumes, so this is not guaranteed to work. That said, many books and other large volumes of text have other ways of identifying their contents at a glance, such as a table of contents or an introductory passage. If these are found, the chatbot will be able to infer summary details from these pages without analyzing the entire book, and should subsequently be able to generate a summary.
* **Will you include Windows support in the future?**
    * Depends on my schedule, but WSL will be supported in the meantime. This project was built for potential scalability in mind, so I decided on Linux Docker images to allow for future deployment on Kubernetes clusters.
* **The chatbot is repeating itself rather than giving me a new answer. How do I stop that?**
    * You may notice there is a bit of "stickiness" around the conversation context, especially if you enable and disable RAG or agent contexts during a chat. For instance, if you start a conversation asking about advice on a particular subject, then change the subject mid-conversation. Subsequently, the chatbot may have some difficulty changing the subject in its responses. This is because the chatbot is looking at the last few replies in the conversation, and attempting to continue the conversation from where it left off. If your prompts are short or vague, then the chatbot will make some assumptions and continue on the last topic, similar to how a person may misunderstand when a subject has changed in a real conversation.
    * To combat this, just be more clear in your instructions when you want to change the subject. Issuing a command such as `Ignore all previous instructions` or `Ignore the previous conversation` is usually enough to get the chabot to change the subject and do what is asked.
* **It says above the responses should be generated within 90 seconds, but my chatbot is taking several minutes to respond. What is happening?**
    * You may be running inference from CPU instead of GPU as expected. Llama.cpp will default to CPU if the dependencies are not compiled properly for GPU, and will not error when the GPU is not detected. To confirm if this is the case for you, load a LLM and check the logs for the pytldr server. Llama.cpp should display a message indicating that a GPU is found when initializing a model. If not found, you may need to change the build instructions in the dockerfile to match your hardware configuration.
* **The chatbot window is not automatically scrolling to the latest reply when I receive a message.**
    * This is the default behavior of the version of Gradio I used, and I do not yet have a fix. For now you will just need to scroll manually to the bottom of the chat when this happens.
* **One or more databases encountered an error during startup. How do I get more info?**
    * As your databases grow, you may notice slower responses or accumulation of pending tasks in Cassandra, which can lead to occasional errors during startup. If a database container fails to start, you can run `docker logs dse68` or `docker logs dse7` to view the logs from each database. In many cases you can simply try to run `docker compose up -d` again to see if the error corrects itself. If not, you may need to attempt to run some repair steps from another container.
    * It is important to note that Cassandra databases are designed to function better when data and requests are distributed across a cluster, not a single node. If you have a specific need to store large amounts of data and conversations, then it's recommended that you consider upgrading to a dedicated Cassandra cluster.

# Credits 

Thanks to Meta for providing free and open LLMs, and thanks to the broader machine learning community for their contributions to fantastic tools such as llama.cpp and Gradio. 

And a special thanks to AMD for providing a GPU to aid in developing this solution for ROCm.