import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
#deprecated from langchain.embeddings import OpenAIEmbeddings
#alsoDeprecated from langchain_community.embeddings import OpenAIEmbeddings #pip install -U langchain-community
from langchain_openai import OpenAIEmbeddings #pip install -U langchain-openai
#deprecated from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI #pip install -U langchain-openai`
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

#TTS
from gtts import gTTS


CHROMA_PATH = "chroma"
load_dotenv() #looks for a .env file in current directory and loads vars 
api_key = os.getenv("OPEN_API_KEY")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("query_text", type=str, help="The query text.")
        args = parser.parse_args()
        query_text = args.query_text
    except:
        query_text="How does Alice meet the Mad Hatter?"


    # Prepare the DB.
    embedding_function = OpenAIEmbeddings(openai_api_key=api_key)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI(openai_api_key=api_key)
    #deprecated response_text = model.predict(prompt)
    response_text = model.invoke(prompt) #newer version of above line

    #TTS
    audio = gTTS(text=f'{response_text.content}', lang='en')
    audio.save('message.mp3')
    #os.system('afplay message.mp3')
    os.system('start message.mp3')
  

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\n \n Sources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()