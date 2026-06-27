import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import anthropic
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(dotenv_path="../.env")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# In-memory storage
pdf_chunks = []
pdf_name = ""


def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def find_relevant_chunks(question, chunks, top_k=4):
    vectorizer = TfidfVectorizer(stop_words='english')
    all_texts = chunks + [question]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    chunk_vectors = tfidf_matrix[:-1]
    question_vector = tfidf_matrix[-1]

    similarities = cosine_similarity(question_vector, chunk_vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    return [chunks[i] for i in top_indices if similarities[i] > 0]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_pdf():
    global pdf_chunks, pdf_name

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files allowed'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    text = extract_text_from_pdf(filepath)
    if not text.strip():
        return jsonify({'error': 'Could not extract text from PDF (may be scanned/image-based)'}), 400

    pdf_chunks = chunk_text(text)
    pdf_name = filename

    return jsonify({
        'message': f'"{filename}" loaded successfully',
        'chunks': len(pdf_chunks),
        'words': len(text.split())
    })


@app.route('/ask', methods=['POST'])
def ask_question():
    if not pdf_chunks:
        return jsonify({'error': 'No PDF loaded yet. Please upload a PDF first.'}), 400

    data = request.json
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    relevant_chunks = find_relevant_chunks(question, pdf_chunks)

    if not relevant_chunks:
        return jsonify({'answer': 'I could not find relevant information in the PDF to answer your question.'})

    context = "\n\n---\n\n".join(relevant_chunks)

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""You are a helpful assistant answering questions about a PDF document.
Use only the context below to answer. If the answer isn't in the context, say "I couldn't find that in the document."

Context from PDF:
{context}

Question: {question}

Answer:"""
            }
        ]
    )

    return jsonify({'answer': message.content[0].text})


@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'loaded': bool(pdf_chunks),
        'filename': pdf_name,
        'chunks': len(pdf_chunks)
    })


if __name__ == '__main__':
    print("Starting PDF Q&A server at http://localhost:8080")
    app.run(debug=True, port=8080)
