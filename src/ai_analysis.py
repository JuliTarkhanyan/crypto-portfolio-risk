import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"


def generate_ai_analysis(
    portfolio_context,
    user_question="",
    model=DEFAULT_MODEL,
):
    """
    Generate AI portfolio analysis using a local Ollama model.
    No OpenAI API key or paid API is required.
    """

    if not portfolio_context:
        raise ValueError("Portfolio context is empty.")

    if not user_question or not user_question.strip():
        user_question = (
            "Give me a concise analysis of this crypto portfolio. "
            "Focus on performance, volatility, risk, diversification, "
            "drawdown, VaR/CVaR, and Monte Carlo results if available."
        )

    system_prompt = """
You are a cryptocurrency portfolio analysis assistant.

Analyze ONLY the data provided in the portfolio context.

Rules:
- Do not invent prices, returns, statistics, or market data.
- Do not use external market information.
- Clearly distinguish historical statistics from Monte Carlo simulations.
- Explain financial metrics in simple language.
- Identify concentration and diversification characteristics.
- Explain important risks and limitations.
- Do not give direct buy or sell instructions.
- Do not claim that Monte Carlo predicts the future.
- If information is missing, say that it is not available.
- Base every conclusion on the supplied data.

Use this structure:

### Overall

### Performance

### Risk

### Diversification

### Monte Carlo

### Key observations

### Limitations
"""

    prompt = f"""
{system_prompt}

PORTFOLIO DATA
==============

{portfolio_context}

END PORTFOLIO DATA
==================

USER QUESTION
=============

{user_question}

Analyze the portfolio using only the supplied information.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=180,
        )

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama is not running. "
            "Start Ollama and make sure llama3.2 is installed."
        )

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollama took too long to respond."
        )

    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Could not connect to Ollama: {e}"
        )

    if response.status_code != 200:
        try:
            error_data = response.json()
            error_message = error_data.get(
                "error",
                response.text
            )
        except Exception:
            error_message = response.text

        raise RuntimeError(
            f"Ollama returned an error: {error_message}"
        )

    try:
        data = response.json()
    except Exception:
        raise RuntimeError(
            "Ollama returned an invalid response."
        )

    answer = data.get(
        "response",
        ""
    ).strip()

    if not answer:
        raise RuntimeError(
            "Ollama returned an empty response."
        )

    return answer