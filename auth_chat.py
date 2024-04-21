import os
from hashlib import sha256

import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings.cohere import CohereEmbeddings
from langchain_community.llms import Cohere
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.vectorstores import Qdrant
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
import psycopg2
os.environ["COHERE_API_KEY"] = "2DNlMKIjntYyI9fflsvWJ9Nqn0cfZSyZUV92J2o6"                                # save the embeddings in a DB that is persistent
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
def connection():
    # See docker command above to launch a postgres instance with pgvector enabled.
    connection = "postgresql+psycopg://nouha:nouha@localhost:5432/rag_db"  # Uses psycopg3!
    collection_name = "my_docs"
    embeddings = CohereEmbeddings()

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
        )
    return vectorstore


#loader = TextLoader("content/cleaned_data_for_finetuning.csv")
#document=loader.load()
#docs=document[0].page_content


# Ajouter un uploader de fichiers dans la sidebar ou dans le corps principal
uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV", type='csv')

# Utiliser le fichier téléchargé s'il existe, sinon utiliser le fichier par défaut
if uploaded_file is not None:
    # Supposons que le fichier téléchargé est un CSV avec les mêmes colonnes que le fichier par défaut
    loader = TextLoader(uploaded_file)
else:
    # Chemin vers le fichier CSV par défaut
    default_file_path = "content/cleaned_data_for_finetuning.csv"
    loader = TextLoader(default_file_path)

# Charger les documents
document = loader.load()
docs = document[0].page_content
text_splitter = CharacterTextSplitter(
    # Set a really small chunk size, just to show.
    separator ='\n\n',
    chunk_size = 1028,
    chunk_overlap  =0,
    length_function = len,
    add_start_index = True,
)
docs = text_splitter.create_documents([docs])
texts = [doc.page_content for doc in docs]



# Define the embeddings model
embeddings = CohereEmbeddings(model = "embed-multilingual-v2.0")

metadata=[{"source": text} for text in texts]
#docs=[Document(page_content=text) for text in texts]
pgvector = connection()
pgvector.add_documents(docs)
RT = pgvector.as_retriever()
# Embed the documents and store in index
#vector_store = Qdrant.from_texts(texts, embeddings, location=":memory:",metadatas=metadata, collection_name="summaries", distance_func="Dot")
#retriever=vector_store.as_retriever()




#create custom prompt for your use case

prompt_template = """Vous êtes Alice, un assistant virtuel spécialisé dans le support SAP, disponible pour répondre aux questions des utilisateurs SAP. 
Répondez aux questions en utilisant les informations fournies. Vous utilisez toujours des formules de politesse telles que "Bonjour" et "Bon après-midi". 
Utilisez les éléments de contexte suivants pour répondre à la question des utilisateurs et citer etapes par etape dans la reponces d'une format comprehensible et detaillée comme dans le document. 
Notez les sources et incluez-les dans la réponse au format : "source1 source2", utilisez "SOURCES" en majuscules quel que soit le nombre de sources. 
Si vous ne connaissez pas la réponse, contentez-vous de dire que "Je ne sais pas", ne cherchez pas à inventer une réponse et aussi si vous avez un grand nombre de tokens .
----------------
{summaries}"""


messages = [
    SystemMessagePromptTemplate.from_template(prompt_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

chain_type_kwargs = {"prompt": prompt}

llm=Cohere(model="command")

#build your chain for RAG+C
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=pgvector.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)
# Interface Streamlit
def chat_interface(chain):
    st.markdown("""
        <style>
            .chat-box {
                background-color: #fafafa;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .question, .answer, .sources {
                margin: 10px 0;
                line-height: 1.5;
            }
            .question-header, .answer-header, .sources-header {
                font-weight: bold;
            }
            .chat-history {
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid #ccc;
            }
        </style>
    """, unsafe_allow_html=True)

    chat_history = st.empty()  # Container to display chat history

    query = st.text_input("Posez votre question ici:", key="query_input")
    if st.button("Envoyer"):
        if "chat_log" not in st.session_state:
            st.session_state["chat_log"] = []
        st.session_state["chat_log"].append({"user": query})
        result = chain(query)
        if result and 'answer' in result:
            st.session_state["chat_log"].append({"bot": result['answer']})
            chat_history.markdown(render_chat_log(st.session_state["chat_log"]), unsafe_allow_html=True)
        else:
            st.write("Désolé, je ne peux pas trouver de réponse à cette question.")

def render_chat_log(chat_log):
    chat_history_html = ""
    for entry in chat_log:
        if "user" in entry:
            chat_history_html += f'<div class="question"><span class="question-header">Question:</span>{entry["user"]}</div>'
        elif "bot" in entry:
            chat_history_html += f'<div class="answer"><span class="answer-header">Réponse:</span>{entry["bot"]}</div>'
    return chat_history_html



















# Simuler une base de données d'utilisateurs avec {username: password_hash}
users = {
    "OCP_user": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"  # Exemple avec le mot de passe "password" hashé
}

# Fonction pour hacher un mot de passe
def hash_password(password):
    """Hash un mot de passe."""
    return sha256(password.encode()).hexdigest()

# Fonction pour vérifier si le mot de passe est correct
def check_password(username, password):
    """Vérifie si le mot de passe est correct."""
    if username in users and users[username] == hash_password(password):
        return True
    return False


# Ajouter du CSS personnalisé pour changer l'image de l'arrière-plan
def set_bg_image():
    st.markdown(
        f"""
        <style> 
        .stApp {{
            background-image: linear-gradient(to bottom, #ADD8E6, #32CD32);
            background-size: cover;
            background-position: center center;
        }}
         /* Styliser la sidebar */
        .css-1v3fvcr {{
            background-color: #f0f2f6; /* Couleur de fond */
            box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.3); /* Ombre portée */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image()  # Appliquer l'image de fond
# Fonction pour créer la page d'authentification
def login_page():
    st.sidebar.title("Connexion")
    username = st.sidebar.text_input("Nom d'utilisateur")
    password = st.sidebar.text_input("Mot de passe", type='password')

    if st.sidebar.button("Se connecter"):
        if check_password(username, password):
            st.session_state['authenticated'] = True
            st.experimental_rerun()  # Rerun the app to update the state
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect!")



# Fonction pour gérer la déconnexion de l'utilisateur
def logout():
    if 'authenticated' in st.session_state:
        del st.session_state['authenticated']  # Supprime l'état authentifié
        st.experimental_rerun()  # Rerun the app to reflect the changes

# Vérifier si l'utilisateur est authentifié
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# Contenu conditionnel basé sur l'authentification
if st.session_state['authenticated']:

    st.success("Vous êtes connecté!")
    # Insérez la logique de votre application ici
    chat_interface(chain)
    if st.button("Déconnecter"):
        logout()
else:
    login_page()

if __name__ == '__main__':
    #v=connection()
    embeddings = CohereEmbeddings(model="embed-multilingual-v2.0")

    metadata = [{"source": text} for text in texts]
    # docs=[Document(page_content=text) for text in texts]
    pgvector = connection()
    pgvector.add_documents(docs)
    RT = pgvector.as_retriever()
    print(RT.invoke("bonjour"))