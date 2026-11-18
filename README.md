# EcoBot — Phase 4, Exercise 3

**Student:** Lilia Maria Ward Bowie  
**Group:** 203238430_8  
**Course:** Natural Language Processing (203238430), UNAD

## Scope and architecture

EcoBot is an environmental educational prototype focused on mangroves, coral bleaching, marine debris and plastic pollution. This implementation keeps the Exercise 2 design: **Qwen/Qwen2.5-0.5B-Instruct** (local language generation), **intfloat/multilingual-e5-small** (query/passage embeddings), **Chroma** (persistent cosine-similarity search), **bounded session memory** (four turns), and **RAG**. The interface uses **Streamlit**.

The dataset in `data/sources.json` contains **five carefully attributed educational paraphrases derived from verifiable NOAA pages**, not copies of the original complete pages or imported PDFs. Each record contains a title, author, URL, last-updated date, topic and source-provenance note. These records were prepared after reviewing the linked public pages on November 18, 2026. NOAA generally permits reuse of NOAA-produced public-domain materials unless otherwise noted; the app is neither affiliated with nor endorsed by NOAA.

The retrieval distance threshold (0.55) is a heuristic; a relevant match does not prove scientific correctness. Citations shown to users are source links, and the answer must be checked against the cited content.

## Local execution

1. Install Python 3.10 or 3.11 (recommended), Git and adequate RAM. CPU operation can be slow.
2. Open a terminal in the directory containing `streamlit_app.py`.
3. Create and activate a virtual environment:
   - Windows: `py -m venv .venv` and `.venv\Scripts\activate`
   - macOS/Linux: `python -m venv .venv` and `source .venv/bin/activate`
4. Run `python -m pip install -r requirements.txt`.
5. Run `streamlit run streamlit_app.py`.
6. When the browser opens, send a question. The first launch downloads the models from Hugging Face, indexes the NOAA-based source records, and can take several minutes.
7. The index is stored in `.chroma_db/`, and changes to the source records trigger rebuilding the index.

Models are loaded locally, with **no paid API** and **no token required**. However, downloading them requires public internet access and sufficient RAM, disk and CPU/GPU resources. The app may not work within very small free hosting limits. Never claim the deployed link is functional until it is tested.

## Free Streamlit Community Cloud deployment

1. Register/sign in at https://github.com and create a repository for EcoBot.
2. Upload `streamlit_app.py`, `ecobot_core.py`, `requirements.txt`, the `data/sources.json` file and `.streamlit/config.toml`. Do not upload `.chroma_db`, environment variables, secrets or local virtual environments.
3. Visit https://share.streamlit.io and connect your GitHub account.
4. Select **Create app** / **Deploy**, choose the repository and branch, and use `streamlit_app.py` as the entrypoint.
5. Wait for dependency installation and model downloads; open **Manage app → Logs** if startup fails.
6. If Community Cloud reports insufficient memory, CPU or disk resources, test locally, try an authorized higher-capacity free runtime, or document the hosting limitation. **Do not claim deployment was successful if it was not.** Keep Qwen and Chroma in the architecture; optimizing weights/quantization would require separate testing.
7. After a successful deployment, copy the actual URL (do not invent it), test at least one Spanish query, one English query, a follow-up and a query with insufficient evidence.

Official documentation: https://docs.streamlit.io/deploy/streamlit-community-cloud

## Evaluation requirements (Exercise 3)

Use the test cases in `evaluation_template.csv` and record the actual model outputs and 1–5 human ratings for relevance, clarity, usefulness and source support. Include test cases for follow-up memory, ambiguity, out-of-coverage questions and local real-time measurements. Compute success rate only after defining a pass condition and completing human assessment. Do not equate the absence of runtime errors with correct answers.

Collect voluntary feedback with `user_feedback_template.csv`. Avoid collecting identifying information. Summarize at least three **observed** systematic issues with evidence, identify possible causes and propose technical improvements. A suggested checklist is in `evaluation_protocol.md`.

## Known limitations

- The open 0.5B-parameter model can hallucinate even when given sources; citations are not a guarantee of correctness.
- The five-source knowledge base is small and consists of authored summaries of NOAA content, not the original documents in full.
- The app has no live ecological data or municipal monitoring.
- The session-memory layer is ephemeral and not shared between users.
- The Chroma directory persists only if the hosting filesystem persists; it can be rebuilt on restart.
- Cold starts, model downloads and CPU inference can be slow.
- No automated external deployment or end-user testing has been carried out as part of this file-generation step.

## APA-style source references

National Oceanic and Atmospheric Administration. (2021, January 4). *What is marine debris?* National Ocean Service. https://oceanservice.noaa.gov/facts/marinedebris.html

National Oceanic and Atmospheric Administration. (2025, April 16). *Plastic marine debris fact sheet*. Marine Debris Program. https://marinedebris.noaa.gov/fact-sheets/plastic-marine-debris-fact-sheet

National Oceanic and Atmospheric Administration. (2026, September 23). *What is coral bleaching?* National Ocean Service. https://oceanservice.noaa.gov/facts/coral_bleach.html

National Oceanic and Atmospheric Administration. (2026, September 23). *What is a mangrove forest?* National Ocean Service. https://oceanservice.noaa.gov/facts/mangroves.html

National Oceanic and Atmospheric Administration. (2026, September 23). *What is marine debris?* National Ocean Service Education. https://oceanservice.noaa.gov/education/tutorial-coastal/marine-debris/md01.html

National Oceanic and Atmospheric Administration. (n.d.). *Frequently asked questions*. National Ocean Service. https://oceanservice.noaa.gov/about/faq.html
