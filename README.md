# Ai-resume-analyzer
# 📄 ATS Resume Checker

A Streamlit app that scores your resume for ATS (Applicant Tracking System) compatibility and tells you exactly how to improve it, powered by Google Gemini Flash.

## Features

- Upload a resume as **PDF or DOCX**
- Optional **job description** box for targeted keyword matching
- Overall **ATS score (0-100)** computed from five weighted categories:
  keywords & relevance (30%), experience & impact (25%), formatting & structure (20%), skills (15%), contact & education (10%)
- Prioritised **improvement suggestions** with example rewrites
- **Missing keywords** and **rewritten bullet points**
- Instant local **quick checks** (email, phone, LinkedIn link, length, standard section headings)
- Downloadable **Markdown report**

## Run locally

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Install and run:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

3. Provide your key in one of these ways:
   - Paste it into the sidebar when the app opens, **or**
   - Create `.streamlit/secrets.toml`:

   ```toml
   GEMINI_API_KEY = "your-key-here"
   # GEMINI_MODEL = "gemini-3.6-flash"   # optional override
   ```

   - Or set an environment variable: `export GEMINI_API_KEY=your-key-here`

> Never commit your API key. `.gitignore` already excludes `.streamlit/secrets.toml` and `.env`.

## Choosing the model

The default is `gemini-3.6-flash`. Google releases new Flash models often, so if you see a "model not found" error, set `GEMINI_MODEL` (secret, env var, or the sidebar "Model" box) to a current Flash model name from the [Gemini models page](https://ai.google.dev/gemini-api/docs/models).

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository (files at the repository root).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** and choose to deploy from an existing repo.
4. Select your repository, branch `main`, and main file path `app.py`.
5. Open **Advanced settings → Secrets** and add:

   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```

6. Click **Deploy**. Your app gets a public `*.streamlit.app` URL.

## Project structure

```
app.py             # the Streamlit app
requirements.txt   # Python dependencies
README.md          # this file
.gitignore         # keeps secrets out of Git
```

## Notes and limitations

- ATS scoring is an **estimate**. Real ATS products differ, so treat the score as guidance, not a guarantee.
- Scanned/image-only resumes cannot be read (and real ATS systems can't read them either). Use a text-based PDF or DOCX.
- Your resume text is sent to Google's Gemini API. Don't upload documents you are not comfortable sharing with that service.
- Maximum file size: 5 MB.
