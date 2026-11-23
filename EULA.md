# WeatherWise AI — End User License Agreement (Draft)


## 1. About the application

WeatherWise AI is a student-built application that answers weather questions and suggests activities based on the forecast. It is intended for educational demonstration and non-commercial use.

## 2. Using WeatherWise AI

You can use the application to check forecasts and explore weather-related activities. Please do not use it for unlawful purposes or enter passwords, payment information, API keys, or other sensitive information into the chat.

WeatherWise AI is not an emergency warning service. If the weather could affect your safety, check official warnings and make your own decision rather than relying on the chatbot alone.

## 3. Forecasts and AI answers

Weather data comes from [Open-Meteo](https://open-meteo.com/). Forecasts can change and may not reflect the exact conditions at a particular time or place.

The chatbot uses OpenAI's **gpt-oss-20b** model through the **Groq API** to interpret questions and explain forecast data. Its answers can contain mistakes, including incorrect dates, comparisons or recommendations. Please check important information before acting on it.

## 4. Data and privacy

To answer a question, the application may send:

- A city name or coordinates to Open-Meteo to retrieve weather information.
- Your question, a small amount of recent chat history, the selected location and relevant forecast data to Groq for AI processing.

The application keeps chat messages in the active Streamlit session. Its logs may include technical information such as response times, token counts and error codes. External providers handle requests under their own privacy policies and retention practices; this draft does not promise that they retain no data.

See [Groq's privacy policy](https://groq.com/privacy-policy) and [Open-Meteo's terms](https://open-meteo.com/en/terms) for more information.

## 5. Service availability

WeatherWise AI requires an internet connection and access to third-party services. It may occasionally return an error, be unavailable or provide incomplete results, for example when an API limit is reached.

## 6. Third-party licenses

The gpt-oss-20b model is released under the **Apache License 2.0**. Open-Meteo weather data is provided under **CC BY 4.0** and must be attributed appropriately. Use of the hosted services remains subject to the relevant providers' terms.

Weather data: [Open-Meteo.com](https://open-meteo.com/), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## 7. Legal review

This academic prototype is provided **as is**, without a promise that its forecasts or AI responses will always be accurate or available. Nothing here removes rights or protections that apply under the law.

Before any public or commercial release, a legal reviewer should confirm the final terms, privacy notices, operator contact details and any applicable data-protection requirements.