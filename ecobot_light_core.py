"""EcoBot Lite: deterministic, source-grounded bilingual retrieval (no hosted LLM).

The hosted demonstration uses only the Python standard library. The full
Qwen + multilingual E5 + Chroma implementation remains in Exercise 2.
"""
from __future__ import annotations
import json
import re
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field

DATA_PATH = Path(__file__).resolve().parent / "data" / "sources.json"


def normalize(value: str) -> str:
    value = unicodedata.normalize('NFKD', value.lower())
    value = ''.join(c for c in value if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9 ]+', ' ', value).strip()


# Both languages are deliberately recognized; no machine inference takes place.
TERMS = {
    'mangroves': ('mangrove', 'manglares', 'manglar', 'raices', 'roots', 'coastal forest', 'bosques costeros'),
    'corals': ('coral', 'corales', 'arrecife', 'arrecifes', 'reef', 'bleaching', 'blanqueamiento', 'algas simbioticas'),
    'plastic': ('plastic', 'plastico', 'plasticos', 'microplastic', 'microplastico', 'microplasticos', 'basura', 'residuos', 'debris', 'litter', 'pollution', 'contaminacion', 'trash', 'waste'),
    'carbon': ('carbon', 'carbono', 'co2', 'dioxido de carbono', 'blue carbon', 'carbono azul', 'climate mitigation', 'captura de carbono'),
}
TOPIC_SOURCES = {
    'mangroves': ['noaa_mangrove'],
    'corals': ['noaa_bleaching'],
    'plastic': ['noaa_plastic_ingestion', 'noaa_plastic', 'noaa_marine_debris', 'noaa_plastic_fact'],
    'carbon': ['noaa_blue_carbon', 'noaa_blue_carbon_detail', 'noaa_mangrove'],
}


def language(question: str) -> str:
    q = question.lower()
    if re.search(r'[áéíóúñ¿¡]', q):
        return 'es'
    spanish = (' que ', ' como ', ' por que ', ' cuales ', ' los ', ' las ', ' para ', ' podemos ', ' explic')
    return 'es' if any(term in ' ' + normalize(q) + ' ' for term in spanish) else 'en'


def is_live_or_exact(question: str) -> bool:
    q = normalize(question)
    live = ['ayer','yesterday','hoy','today','ahora','right now','esta manana','last night',
            'exacta','exacto','exact','2026','this morning','cuantos hectareas','nivel del mar',
            'sea level','medicion','measurement','cobertura','coverage']
    return any(s in q for s in live)


def wants_child_explanation(q: str) -> bool:
    n=normalize(q)
    return any(t in n for t in ['nino', 'nina', '10 anos', '10 years', 'child', 'kid', 'simplif', 'mas facil', 'simple words'])


def is_followup(q: str) -> bool:
    n=' '+normalize(q)+' '
    return any(t in n for t in [' eso ', ' los ', ' las ', ' protegerlos ', ' them ', ' those ', ' it ', ' that ', ' what about ', ' y como ', ' and how '])


def detected_topics(q: str) -> list[str]:
    n=' '+normalize(q)+' '
    result=[]
    for topic, words in TERMS.items():
        if any(' '+normalize(w)+' ' in n for w in words):
            result.append(topic)
    return result


@dataclass
class Memory:
    max_turns: int = 5
    turns: list = field(default_factory=list)
    last_topic: str | None = None
    last_answer: str = ''
    def clear(self):
        self.turns.clear()
        self.last_topic = None
        self.last_answer = ''
    def remember(self, q, a, topic):
        self.turns.append({'question':q, 'answer':a, 'topic':topic})
        self.turns=self.turns[-self.max_turns:]
        if topic:
            self.last_topic=topic
        self.last_answer=a


ANSWERS = {
'mangroves': {
 'es': 'Los manglares son bosques costeros de árboles y arbustos adaptados a ambientes salobres. Sus raíces brindan refugio y alimento a peces y otros organismos, ayudan a retener sedimentos y contribuyen a estabilizar la costa frente a las olas y la erosión. Su función depende de las condiciones de cada lugar y del estado del ecosistema.',
 'en': 'Mangroves are coastal forests of salt-tolerant trees and shrubs. Their roots provide habitat for fish and other organisms, trap sediments, and help stabilize shorelines against waves and erosion. Their benefits vary with local conditions and ecosystem health.'},
'corals': {
 'es': 'El blanqueamiento coralino ocurre cuando el estrés ambiental altera la relación entre el coral y sus algas simbióticas, que aportan nutrientes y color. El calentamiento del agua es una causa importante; también influyen otros factores ambientales. Un coral blanqueado no necesariamente está muerto, pero puede morir si el estrés continúa.',
 'en': 'Coral bleaching occurs when environmental stress disrupts the relationship between corals and their symbiotic algae, which provide nutrients and color. High seawater temperatures are a major cause. Bleached coral is not necessarily dead, but prolonged stress can lead to mortality.'},
'plastic': {
 'es': 'La basura plástica puede llegar al mar desde tierra por la lluvia, ríos y drenajes. Los animales marinos pueden ingerir fragmentos o enredarse en residuos y artes de pesca. La ingestión puede provocar lesiones digestivas, obstrucciones y desnutrición. Para prevenir estos impactos, es útil reducir los desechables, disponer correctamente de los residuos y participar en limpiezas seguras.',
 'en': 'Plastic waste can reach the ocean through rivers, rainwater and drains. Marine animals may ingest fragments or become entangled in discarded debris and fishing gear. Ingestion may cause digestive injuries, blockage and malnutrition. Preventive actions include reducing single-use plastics, proper disposal and safe community cleanups.'},
'carbon': {
 'es': 'Los manglares almacenan carbono azul: capturan dióxido de carbono y guardan carbono en su vegetación y, especialmente, en los suelos y sedimentos costeros. Estos depósitos pueden mantenerse durante largos períodos. Proteger los manglares ayuda a conservar ese carbono y a sostener la biodiversidad. La cantidad exacta almacenada depende de cada ecosistema.',
 'en': 'Mangroves store blue carbon: they capture carbon dioxide and retain carbon in vegetation and especially in coastal soils and sediments. This storage may persist for long periods. Protecting mangroves helps conserve these carbon stocks and biodiversity. The amount stored varies by ecosystem.'},
}
CHILD = {
 'carbon': {
  'es': 'Imagina que los manglares son como una alcancía natural: toman parte del carbono del aire y lo guardan en sus raíces, árboles y, sobre todo, en el suelo. Cuidarlos ayuda a mantener ese carbono almacenado.',
  'en': 'Think of mangroves as a natural piggy bank: they take in some carbon from the air and keep it in their trees, roots and especially the soil. Protecting them helps keep that carbon stored.'},
 'mangroves': {
  'es': 'Los manglares son árboles que viven cerca del mar. Sus raíces parecen una casa para peces pequeños y ayudan a que las olas no se lleven tanta tierra de la costa.',
  'en': 'Mangroves are trees that grow near the sea. Their roots are like a home for small fish and help prevent waves from washing away coastal soil.'},
 'corals': {
  'es': 'Los corales viven con algas diminutas que les dan alimento y color. Si el agua se calienta demasiado, pueden perder esas algas y verse blancos. Eso se llama blanqueamiento coralino.',
  'en': 'Corals live with tiny algae that give them food and color. If the water gets too warm, corals may lose those algae and turn white. This is called coral bleaching.'},
 'plastic': {
  'es': 'Un animal marino puede confundir un pedazo de plástico con comida. El plástico puede hacerle daño a su estómago o impedir que coma bien. Por eso es importante no dejar basura cerca del mar.',
  'en': 'A sea animal can mistake plastic for food. Plastic may hurt its stomach or prevent it from eating properly. That is why keeping trash out of the ocean matters.'},
}
PROTECT = {
 'mangroves': {'es':'Para proteger los manglares, evita arrojar basura o dañarlos, apoya su conservación y restauración con organizaciones locales y ayuda a mantener limpios los canales y costas. Las acciones de restauración deben adaptarse al lugar y contar con orientación técnica.',
               'en':'To protect mangroves, avoid dumping waste or damaging trees, support local conservation and carefully planned restoration, and help keep coastal waterways clean. Restoration should follow local ecological guidance.'},
 'corals': {'es':'Para proteger los arrecifes, evita tocar o pisar corales, no dejes residuos, respeta las normas de las áreas protegidas y apoya acciones para reducir la contaminación y el calentamiento global.',
            'en':'To protect reefs, avoid touching or standing on corals, prevent litter, follow marine protected area rules, and support actions that reduce pollution and climate-related stress.'},
 'plastic': {'es':'Para reducir la contaminación marina, evita los plásticos innecesarios de un solo uso, separa y deposita correctamente los residuos y participa en limpiezas costeras organizadas de forma segura.',
             'en':'To reduce marine pollution, avoid unnecessary single-use plastics, dispose of waste correctly, and join safely organized coastal cleanups.'},
 'carbon': {'es':'Proteger los bosques de manglar y sus suelos ayuda a conservar el carbono azul almacenado. También es importante evitar la degradación del hábitat y apoyar su manejo responsable.',
            'en':'Protecting mangrove forests and their soils helps preserve blue carbon stocks. Preventing habitat degradation and supporting responsible management are important.'},
}


def retrieve(topic: str, by_id: dict, q: str) -> list[dict]:
    ids=TOPIC_SOURCES.get(topic, [])
    n=normalize(q)
    if topic=='plastic':
        if any(x in n for x in ('ingest','inger','animal','tortuga','turtle')):
            ids=['noaa_plastic_ingestion','noaa_plastic','noaa_marine_debris']
        elif any(x in n for x in ('educat','escuela','school','classroom','actividad')):
            ids=['noaa_plastic_fact','noaa_plastic','noaa_marine_debris']
        else:
            ids=['noaa_plastic','noaa_marine_debris','noaa_plastic_fact']
    return [by_id[i] for i in ids[:2] if i in by_id]


def answer(question: str, memory: Memory, documents: list[dict] | None = None) -> dict:
    docs=documents if documents is not None else json.loads(DATA_PATH.read_text(encoding='utf-8'))
    by_id={d['id']:d for d in docs}
    lang=language(question)
    topics=detected_topics(question)
    # Select carbon over mangroves when both are present.
    topic='carbon' if 'carbon' in topics else (topics[0] if topics else None)
    if not topic and memory.last_topic and (is_followup(question) or wants_child_explanation(question)):
        topic=memory.last_topic
    if is_live_or_exact(question):
        message = ('No dispongo de mediciones locales verificadas ni datos ambientales en tiempo real para responder con una cifra exacta. Consulta las autoridades ambientales y los conjuntos de datos oficiales del lugar.' if lang=='es' else 'I do not have verified local measurements or live environmental data to provide an exact number. Please consult relevant official monitoring datasets or environmental authorities.')
        memory.remember(question,message,topic)
        return {'answer':message, 'sources':[], 'status':'insufficient_evidence', 'topic':topic}
    if not topic:
        message=('Puedo explicar temas sobre manglares, blanqueamiento coralino, carbono azul y residuos plásticos. ¿Sobre cuál te gustaría preguntar?' if lang=='es' else 'I can explain mangroves, coral bleaching, blue carbon and plastic marine debris. Which topic would you like to explore?')
        memory.remember(question,message,None)
        return {'answer':message,'sources':[],'status':'clarification','topic':None}
    n=normalize(question)
    if wants_child_explanation(question):
        response=CHILD[topic][lang]
    elif any(k in n for k in ('proteg','protect','conserv','reduc','reduce','prevent','evitar','cuidar','help them')):
        response=PROTECT[topic][lang]
    elif 'plastic'==topic and any(k in n for k in ('actividad','activity','school','escuela','educat','teach','clase')):
        response=('Para una actividad escolar, el grupo puede contar los residuos plásticos de uso diario, identificar cómo podrían llegar al mar y diseñar un plan para reducirlos. Realiza cualquier limpieza con supervisión y medidas de seguridad.' if lang=='es' else 'For a classroom activity, learners can count daily single-use plastic items, discuss how litter reaches the ocean, and propose a waste-reduction plan. Any cleanup should be supervised and safe.')
    else:
        response=ANSWERS[topic][lang]
    sources=retrieve(topic,by_id,question)
    formatted=[{'id':d['id'],'title':d['title'],'url':d['url'],'author':d['author'],'source_type':d.get('source_type','source_based_paraphrase')} for d in sources]
    memory.remember(question,response,topic)
    return {'answer':response,'sources':formatted,'status':'retrieval_and_curated_summary','topic':topic}
