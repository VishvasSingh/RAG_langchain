from langchain_community.llms import Ollama
from flask import Flask, request

llm = Ollama(model='llama3')
print(llm.invoke('Tell me a bad joke'))

app = Flask(__name__)


@app.route("/ai", methods=["POST"])
def ai_post():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get('query')

    print(f"query: {query}")

    response = llm.invoke(query)

    return response



def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()



