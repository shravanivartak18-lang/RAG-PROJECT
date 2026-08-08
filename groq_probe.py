import requests

def probe_urls(urls):
    for url in urls:
        try:
            r = requests.get(url, timeout=10)
            print(url, r.status_code, repr(r.text[:200]))
        except Exception as e:
            print(url, 'ERR', type(e).__name__, e)

urls = [
    'https://api.groq.com/v1/embeddings',
    'https://api.groq.com/v1/generate',
    'https://api.groq.com/v1',
    'https://api.groq.com',
    'https://api.groq.dev/v1/embeddings',
    'https://api.groq.dev/v1/generate',
    'https://api.groq.dev/v1',
    'https://api.groq.cloud/v1/embeddings',
    'https://api.groq.cloud/v1/generate',
    'https://api.groq.cloud/v1',
    'https://api.groq.ai/v1/embeddings',
    'https://api.groq.ai/v1/generate',
    'https://api.groq.ai/v1',
]

probe_urls(urls)
