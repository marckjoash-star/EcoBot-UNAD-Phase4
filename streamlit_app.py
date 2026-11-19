"""EcoBot Lite web interface. Does not run a language model on the server."""
import streamlit as st
from ecobot_light_core import Memory, answer

st.set_page_config(page_title='EcoBot Lite | Environmental Education', page_icon='🌿', layout='centered')
st.title('🌿 EcoBot')
st.caption('Lightweight environmental education assistant · Source-grounded retrieval')
st.info('This public demo uses document retrieval and pre-reviewed educational summaries. It does **not** run the Qwen language model. The full LLM + embeddings + Chroma RAG prototype is documented separately in Exercise 2.')
with st.sidebar:
    st.header('About EcoBot')
    st.write('Explore mangroves, blue carbon, coral bleaching and marine plastic pollution in Spanish or English.')
    st.warning('Educational prototype, not live environmental monitoring or professional advice. Review the original sources for high-stakes decisions.')
    st.caption('Knowledge base: eight attributed educational paraphrases based on NOAA material; not verbatim full-page ingestion.')
    if st.button('Clear conversation',use_container_width=True):
        st.session_state.messages=[]
        st.session_state.memory=Memory()
        st.rerun()
    st.divider()
    st.caption('UNAD · Natural Language Processing · Group 203238430_8')
    st.caption('Lilia Maria Ward Bowie · Phase 4')

if 'messages' not in st.session_state:
    st.session_state.messages=[]
if 'memory' not in st.session_state:
    st.session_state.memory=Memory()

if not st.session_state.messages:
    st.markdown('**Example questions:**')
    st.caption('• ¿Por qué son importantes los manglares?')
    st.caption('• What causes coral bleaching?')
    st.caption('• ¿Qué es el carbono azul?')

for item in st.session_state.messages:
    with st.chat_message(item['role']):
        st.markdown(item['content'])
        if item.get('sources'):
            with st.expander('Documentary sources'):
                for source in item['sources']:
                    st.markdown(f"[{source['title']}]({source['url']}) · {source['author']}")

question=st.chat_input('Ask a question / Haz una pregunta')
if question:
    st.session_state.messages.append({'role':'user','content':question})
    with st.chat_message('user'):
        st.markdown(question)
    try:
        result=answer(question,st.session_state.memory)
    except Exception as exc:
        st.error('Could not retrieve a response. Please try again.')
        st.session_state.messages.pop()
        st.stop()
    with st.chat_message('assistant'):
        st.markdown(result['answer'])
        if result['sources']:
            with st.expander('Documentary sources'):
                for source in result['sources']:
                    st.markdown(f"[{source['title']}]({source['url']}) · {source['author']}")
        st.caption('Response method: '+result['status'])
    st.session_state.messages.append({'role':'assistant','content':result['answer'],'sources':result['sources']})
