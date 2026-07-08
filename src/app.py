"""
app.py
Minimal Flask web interface for the Maps RAG search tool.
Run with: python src/app.py
"""

from flask import Flask, request, jsonify, render_template_string
from src.query import answer_query

app = Flask(__name__)

# Minimal single-page UI — replace with a proper template later if desired
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>My Saved Restaurants</title>
    <style>
        body { font-family: sans-serif; max-width: 700px; margin: 60px auto; padding: 0 20px; }
        input { width: 100%; padding: 12px; font-size: 16px; box-sizing: border-box; }
        button { margin-top: 10px; padding: 10px 20px; font-size: 16px; cursor: pointer; }
        #result { margin-top: 30px; white-space: pre-wrap; line-height: 1.6; }
        #spinner { display: none; margin-top: 20px; color: gray; }
    </style>
</head>
<body>
    <h2>Search My Saved Restaurants</h2>
    <input type="text" id="query" placeholder='e.g. "Italian places open now that aren\'t too expensive"' />
    <button onclick="search()">Search</button>
    <div id="spinner">Searching...</div>
    <div id="result"></div>

    <script>
        async function search() {
            const query = document.getElementById('query').value;
            if (!query) return;
            document.getElementById('result').innerText = '';
            document.getElementById('spinner').style.display = 'block';

            const response = await fetch('/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
            const data = await response.json();
            document.getElementById('spinner').style.display = 'none';
            document.getElementById('result').innerText = data.answer;
        }

        // Allow Enter key to trigger search
        document.getElementById('query').addEventListener('keydown', e => {
            if (e.key === 'Enter') search();
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"answer": "Please enter a search query."})

    try:
        answer = answer_query(query)
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"answer": "Sorry, an error occurred while processing your request. Please try again later."}), 500


if __name__ == "__main__":
    app.run(debug=True)
