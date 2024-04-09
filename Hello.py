
import streamlit as st
import time
import os
from PyPDF2 import PdfReader
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import TextLoader, PyPDFLoader, PyMuPDFLoader
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

VERBOSE = True
MAX_TOKENS = 2048
MODEL_CONTEXT_WINDOW = 8192
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 500
HUGGINGFACEHUB_API_TOKEN = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

STYLES = {
    "List": {
        "style": "Return your response as numbered list which covers all the paths that the research paper is trying to accomplish.",
        "trigger": "NUMBERED LIST SUMMARY WITH KEY POINTS AND FACTS",
    },
    "One sentence": {
        "style": "Return your response as one sentence which covers what this research paper is trying to accomplish and their results.",
        "trigger": "ONE SENTENCE SUMMARY",
    },
    "Consise": {
        "style": "Return your response as concise summary which covers what this research paper is trying to accomplish and what technique/idea they are using to accomplish it and their results. Give a segregated response",
        "trigger": "CONCISE SUMMARY",
    },
    "Detailed": {
        "style": "Return your response as detailed summary which covers in detail about the research paper all about it.",
        "trigger": "DETAILED SUMMARY",
    },
    "Custom": {
        "style": "Custom",
        "trigger": "CUSTOM SUMMARY",
    },
}

LANGUAGES = ["Default", "English", "Hindi", "Japanese"]

# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceEndpoint(repo_id=repo_id, token=HUGGINGFACEHUB_API_TOKEN)

combine_prompt_template = """
Write a summary of the following Research Paper/text delimited by tripple backquotes.Give a segregated Response.
>>IGNORE References in the research paper/article.
{style}

```{content}```

{trigger} {in_language}:
"""

map_prompt_template = """
Write a concise summary of the following Research Paper/text which covers the main points and methods and conclusion:
>>IGNORE References in the research paper/article.
{text}

CONCISE SUMMARY {in_language}:
"""

def summarize_base(llm, content, style, language, custom_prompt=None):
    """Summarize whole content at once. The content needs to fit into the model's context window."""
    prompt = PromptTemplate.from_template(
        combine_prompt_template
    ).partial(
        style=STYLES[style]["style"],
        trigger=STYLES[style]["trigger"],
        in_language=f"in {language}" if language != "Default" else "",
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=VERBOSE)
    output = chain.run(content)
    return output

def summarize_map_reduce(llm, content, style, language, custom_prompt=None):
    """Summarize content potentially larger than the model's context window using map-reduce approach."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    split_docs = text_splitter.create_documents([content])
    print(
        f"Map-Reduce content splits ({len(split_docs)} splits): {[len(sd.page_content) for sd in split_docs]}")
    map_prompt = PromptTemplate.from_template(
        map_prompt_template
    ).partial(
        in_language=f"in {language}" if language != "Default" else "",
    )
    
    combine_prompt = PromptTemplate.from_template(
        combine_prompt_template
    ).partial(
        style=STYLES[style]["style"],
        trigger=STYLES[style]["trigger"],
        in_language=f"in {language}" if language != "Default" else "",
    )

    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        combine_document_variable_name="content",
        verbose=VERBOSE,
    )

    output = chain.run(split_docs)
    return output

def load_input_file(input_file):
    if not input_file:
        return None
    start_time = time.perf_counter()
    if input_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(input_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()+"\n\n\n"
        return text
    docs = TextLoader(input_file.name).load()
    end_time = time.perf_counter()
    print(f"Input file load time {round(end_time - start_time, 1)} secs")
    return docs[0].page_content

def summarize_text(content, style, language, custom_prompt=None):
    content_tokens = llm.get_num_tokens(content)

    print("Content length:", len(content))
    print("Content tokens:", content_tokens)
    print("Content sample:\n" + content[:200] + "\n\n")

    info = f"Content length: {len(content)} chars, {content_tokens} tokens."

    # Keep part of context window for models output & some buffer for the prompt.
    base_threshold = MODEL_CONTEXT_WINDOW - MAX_TOKENS - 256

    start_time = time.perf_counter()

    if (content_tokens < base_threshold):
        info += "\n"
        info += "Using summarizer: base"

        print("Using summarizer: base")
        summary = summarize_base(llm, content, style, language, custom_prompt)
    else:
        info += "\n"
        info += "Using summarizer: map-reduce"

        print("Using summarizer: map-reduce")
        summary = summarize_map_reduce(llm, content, style, language, custom_prompt)

    end_time = time.perf_counter()

    print("Summary length:", len(summary))
    print("Summary tokens:", llm.get_num_tokens(summary))
    print("Summary:\n" + summary + "\n\n")

    info += "\n"
    info += f"Processing time: {round(end_time - start_time, 1)} secs."
    info += "\n"
    info += f"Summary length: {llm.get_num_tokens(summary)} tokens."

    print("Info", info)
    return summary, info

st.title("Summarization Tool")
st.markdown("Drop a file or paste text to summarize it!")

input_file = st.file_uploader("Drop a file here", type=["txt", "pdf"])
input_text = st.text_area("Text to summarize", "", height=300)
style_radio = st.radio("Response style", list(STYLES.keys()))
language_dropdown = st.selectbox("Response language", LANGUAGES)

if style_radio == "Custom":
    custom_prompt = st.text_area("Custom Prompt", "")
    STYLES['Custom']["style"]=custom_prompt
else:
    custom_prompt = None

if input_file is None:
    input_text = input_text
else:
    input_text = load_input_file(input_file)

if st.button("Generate Summary"):
    summary, info = summarize_text(input_text, style_radio, language_dropdown, custom_prompt)
    st.markdown("## Summary")
    st.text_area("Summary", summary, height=300)
    st.text_area("Diagnostic info", info, height=100)
