# WeatherWise AI

WeatherWise AI is a Streamlit chatbot that answers weather questions and suggests activities using forecast data. It was developed for **ITC 6226 — AI Applications Development** at The American College of Greece.

## How it works

- **Open-Meteo** provides city lookup and hourly weather forecasts (today and the next 10 days).
- **Groq (`openai/gpt-oss-20b`)** interprets the requested date or time, then generates an answer grounded in the selected forecast.
- **Pandas** filters the forecast and prepares four-hour or daily summaries.
- **Streamlit** provides the chat interface. The city is selected in the sidebar; comparisons between multiple cities are not supported.
- **Production uses an Nginx reverse proxy and a Nuitka-compiled `app.py` module, launched through a small Python entrypoint..

## Requirements

- Docker Desktop with Docker Compose
- A Groq API key

## Configuration

In the project directory, create a local file named `.env`:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=openai/gpt-oss-20b
```

Replace the placeholder with your own key. **Never include `.env` or any credentials in Git or submission ZIP files.** The provided Docker Compose configuration supplies this key to the application container.

## Run the Production application

From the project directory:

```bash
docker compose -f compose.yaml up -d --build
```

Open **http://localhost:8501** in your browser. Enter a city in the sidebar, click **Use this city**, and ask a weather-related question.

Check container status or view application logs:

```bash
docker compose -f compose.yaml ps
docker compose -f compose.yaml logs -f app
```

Check the health endpoint:

```bash
curl -fsS http://localhost:8501/_stcore/health
```

Stop Production:

```bash
docker compose -f compose.yaml down
```

## Run the Development environment

Stop Production first because the two environments use the same host port.

```bash
docker compose -f compose.yaml down
docker compose -f compose.dev.yaml up -d --build
```

Open **http://localhost:8501**. Run the offline unit tests with:

```bash
docker compose -f compose.dev.yaml run --rm app \
  python -m unittest discover -s tests -v
```

Stop Development:

```bash
docker compose -f compose.dev.yaml down
```

## Production checks

To verify that Nuitka's compiled module loads:

```bash
docker compose -f compose.yaml exec app python -c \
  "import app; assert hasattr(app, '__compiled__'); print('Nuitka PASS')"
```

To validate the Nginx configuration:

```bash
docker compose -f compose.yaml exec proxy nginx -t
```

Dependency audits, container vulnerability scans, and profiling outputs are saved in `scans/`.

## Limitations and data handling

- Forecasts are predictions and can change. AI-generated answers may contain errors; verify important decisions independently.
- The chatbot uses the **city selected in the sidebar**, not a different city named in a chat message.
- It does not have every possible weather variable (for example, a UV index). It is not an emergency weather-alert service.
- The selected city and coordinates are used to retrieve data from Open-Meteo. Weather questions, limited conversation context, and relevant forecast summaries are sent to Groq for inference.
- Hosted Groq inference uses the model ID `openai/gpt-oss-20b`; the exact provider-served model-weight revision cannot be independently pinned by this application.
- The Open-Meteo free API is used for this non-commercial educational project; other uses must comply with the provider's terms.

Weather data: [Open-Meteo](https://open-meteo.com/) ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).

See `EULA.md` for the draft terms of use.
