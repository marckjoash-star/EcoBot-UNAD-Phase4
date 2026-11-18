"""EcoBot environmental education chatbot interface (Exercise 3)."""
import streamlit as st
from ecobot_core import EcoBot, Memory, CONFIG

st.set_page_config(page_title='EcoBot | Coastal Conservation',page_icon='🌿',layout='centered')
st.title('🌿 EcoBot')
st.caption('Environmental education | Coastal ecosystem conservation | Open-source RAG chatbot')

with st.sidebar:
    st.header('About EcoBot')
    st.write('Ask educational questions about mangroves, coral bleaching, marine debris and plastic pollution. Questions in Spanish or English are supported.')
    st.warning('Educational prototype. Responses may contain errors. This assistant does not provide current measurements, professional environmental assessments or legal advice.')
    st.caption('Sources: NOAA National Ocean Service and NOAA Marine Debris Program. Educational source-based summaries; not the complete webpages.')
    if st.button('Clear conversation',use_container_width=True):
        st.session_state.history=[]
        st.session_state.memory=Memory(max_turns=CONFIG.memory_turns)
        st.rerun()
    st.divider()
    st.caption('UNAD · Natural Language Processing · Group 203238430_8')
    st.caption('Lilia Maria Ward Bowie · Phase 4')

if 'history' not in st.session_state:
    st.session_state.history=[]
if 'memory' not in st.session_state:
    st.session_state.memory=Memory(max_turns=CONFIG.memory_turns)

@st.cache_resource(show_spinner='Loading the open-source models and NOAA-based knowledge base…')
def init_bot():
    return EcoBot()

try:
    bot=init_bot()
except Exception as exc:
    st.error('The chatbot could not initialize. The hosting environment may lack RAM, disk space, network access or compatible dependencies.')
    with st.expander('Technical information'):
        st.code(f'{type(exc).__name__}: {exc}')
    st.info('See README.md for local execution and free deployment options. This application requires the Qwen and multilingual E5 model downloads.')
    st.stop()

if not st.session_state.history:
    st.markdown('**Try a question:**')
    for p in ['¿Por qué son importantes los manglares?', 'What causes coral bleaching?', '¿Cómo reducir la basura plástica marina?']:
        st.caption('• '+p)

for item in st.session_state.history:
    with st.chat_message(item['role']):
        st.markdown(item['content'])
        if item.get('sources'):
            with st.expander('Documentary sources'):
                for s in item['sources']:
                    st.markdown(f"**[{s['label']}] {s['title']}** — [NOAA source]({s['url']})")

q=st.chat_input('Ask an environmental question / Haz una pregunta ambiental')
if q:
    st.session_state.history.append({'role':'user','content':q})
    with st.chat_message('user'):
        st.markdown(q)
    with st.chat_message('assistant'):
        with st.spinner('Searching evidence and generating a response…'):
            try:
                result=bot.respond(q,st.session_state.memory)
            except Exception as exc:
                st.error('An error occurred while generating the response. Please try again.')
                with st.expander('Technical information'):
                    st.code(f'{type(exc).__name__}: {exc}')
                # Drop the user turn so the conversational history remains consistent.
                st.session_state.history.pop()
                st.stop()
        st.markdown(result['answer'])
        if result['sources']:
            with st.expander('Documentary sources'):
                for s in result['sources']:
                    st.markdown(f"**[{s['label']}] {s['title']}** — [NOAA source]({s['url']})")
        st.caption('Status: '+result['status'] + ' · Generation: '+str(result['duration_sec'])+' s')
    st.session_state.history.append({'role':'assistant','content':result['answer'],'sources':result['sources']})
