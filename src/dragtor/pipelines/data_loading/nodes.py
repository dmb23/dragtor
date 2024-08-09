import requests
from kedro.framework.session import get_current_session


def load_jina_reader(addr: str) -> requests.Response:
    jina_url = f"https://r.jina.ai/{addr}"
    session = get_current_session()
    context = session.load_context()
    credentials = context._get_config_credentials()
    headers = {"Authorization": f"Bearer {credentials['jina']['api_key']}"}

    response = requests.get(jina_url, headers=headers)

    return response.text
