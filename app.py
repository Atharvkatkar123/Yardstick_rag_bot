from flask import Flask, request, jsonify, render_template_string
import google.generativeai as genai
import os
import json
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import datetime
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_AIP_KEY"))

limiter = Limiter(key_func=get_remote_address,
                  app=app,
                  default_limits=["30 per hour"])

documents = None
doc_embeddings = None

def load_documents():
    #Load Precomputed Embeddings
    global documents, doc_embeddings
    if documents is None:
        print("Loading documents...")
        with open("yardstick_docs.json") as f:
            documents = json.load(f)
        try:
            with open("yardstick_embeddings.json") as f:
                doc_embeddings = json.load(f)
            print(f"Loaded {len(documents)} documents with embeddings")
        except FileNotFoundError:
            print("No embeddings file found, using keyword search")
            doc_embeddings = None
        
def consine_similarity(a,b):
    #cosine similarity
    import math
    dot_product = sum(x*y for x,y in zip(a,b))
    magnitude_a = math.sqrt(sum(x*x for x in a))
    magnitude_b = math.sqrt(sum(x*x for x in b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    return dot_product/(magnitude_a*magnitude_b)

def get_embedding(text):
    #google api
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print("Embedding error: {e}")
        return None
    
def semantic_search(query,k=5):
    #SEARCH using embeddings
    load_documents()
    query_emb = get_embedding(query)
    if query_emb is None or doc_embeddings is None:
        return keyword_search(query,k)
    similarity = []
    for i,doc_emb in enumerate(doc_embeddings):
        sim = consine_similarity(query_emb,doc_emb)
        similarity.append((sim,1))
        
    similarity.sort(reverse=True)
    return [documents[i] for _, i in similarity[:k]]

def keyword_search(query, k=10):
    load_documents()
    keywords = query.lower().split()
    scored = []
    
    for i,doc in enumerate(documents):
        doc_lower = doc.lower()
        score = sum(doc_lower.count(kw) for kw in keywords)
        if query.lower() in doc_lower:
            score += 100
        if score > 0:
            scored.append((score,i))
    
    scored.sort(reverse=True)
    return [documents[i] for _, i in scored[:k]]


def generate_answer(query):
    """Generate answer using semantic search + Gemini"""
    # Use semantic search if embeddings available, else keyword
    if doc_embeddings:
        relevant_docs = semantic_search(query, k=5)
    else:
        relevant_docs = keyword_search(query, k=10)
    
    if not relevant_docs:
        return "I don't have information about that. Please ask about doctors, facilities, or hospital services."
    
    context = '\n\n'.join(relevant_docs)
    
    prompt = f"""Yardstick AI assistant: helpful, professional, concise.

Answer from context (2-3 sentences, benefit-focused). If missing info: acknowledge + offer free strategy call. Pricing: "Depends on needs - what's your use case?" Technical: redirect to team. Contact: contact@yardstick.live | +917891053001

Never fabricate. Stay positive.
{context}

User QUESTION: {query}

YOUR ANSWER:"""
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return "I'm having trouble processing your request. Please try again in a moment."

@app.route('/health')
def health():
    return jsonify({
        'status': 'alive',
        'docs_loaded': documents is not None,
        'embeddings_loaded': doc_embeddings is not None,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/ping')
def ping():
    return 'pong', 200

@app.route('/')
def home():
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Yardstick - AI Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --purple-dark: #22044D;
            --purple-mid: #720ABC;
            --purple-accent: #580063;
            --pink-muted: #47203C;
            --pink-bright: #EA2BAE;
            --bg-dark: #000000;
            --bar-color: #1C151D;
            --text-white: #FFFFFF;
            --text-gray: #B8B8B8;
            --shadow: rgba(71, 32, 60, 0.3);
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: radial-gradient(circle at top right, var(--purple-accent) 0%, var(--bg-dark) 40%, var(--bg-dark) 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* Header - Fixed at top */
        .chat-header {
            background: var(--bar-color);
            color: var(--text-white);
            padding: 15px 20px;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
            border-bottom: 1px solid rgba(88, 0, 99, 0.3);
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 100;
        }

        .logo-container {
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .logo-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .header-text h1 {
            font-size: 1.3rem;
            font-weight: 400;
            color: var(--text-white);
        }

        /* Messages Area - Centered with max-width */
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 80px 20px 120px 20px;
            background: var(--bg-dark);
            min-height: 500px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .messages-container {
            width: 100%;
            max-width: 700px;
        }

        .message {
            display: flex;
            margin-bottom: 20px;
            animation: slideIn 0.3s ease;
            width: 100%;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message.user {
            justify-content: flex-end;
        }

        .message.bot {
            justify-content: flex-start;
        }

        .message-content {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            max-width: 85%;
        }

        .message.user .message-content {
            flex-direction: row-reverse;
        }

        .message-icon {
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            flex-shrink: 0;
            margin-top: 2px;
            overflow: hidden;
        }

        .message.bot .message-icon {
            background: linear-gradient(135deg, var(--pink-muted) 0%, var(--purple-mid) 100%);
            padding: 5px;
        }

        .message.user .message-icon {
            background: rgba(88, 0, 99, 0.6);
        }

        .message-icon img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            border-radius: 8px;
        }


        .message.bot .message-icon {
            background: linear-gradient(135deg,#000000 0%, #000000 100%);
        }

        .message.user .message-icon {
            background: rgba(88, 0, 99, 0.6);
        }

        .message-bubble {
            padding: 12px 18px;
            border-radius: 18px;
            font-size: 0.95rem;
            line-height: 1.6;
            word-wrap: break-word;
            display: inline-block;
            max-width: 100%;
        }

        .message.bot .message-bubble {
            background: var(--bar-color);
            color: var(--text-white);
            border: 1px solid rgba(88, 0, 99, 0.4);
            border-radius: 18px 18px 18px 4px;
        }

        .message.user .message-bubble {
            background: linear-gradient(135deg, #000000 0%, #000000 100%);
            color: var(--text-white);
            border-radius: 18px 18px 4px 18px;
        }

        /* Typing Indicator */
        .typing-indicator {
            display: none;
            align-items: center;
            gap: 12px;
            width: 100%;
            max-width: 800px;
            padding: 0 20px;
            margin: 0 auto;
        }

        .typing-indicator.active {
            display: flex;
        }

        .typing-indicator-icon {
            width: 35px;
            height: 35px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--pink-muted) 0%, var(--purple-mid) 100%);
            flex-shrink: 0;
        }

        .typing-dots {
            display: flex;
            align-items: center;
            padding: 12px 18px;
            background: var(--bar-color);
            border: 1px solid rgba(88, 0, 99, 0.4);
            border-radius: 18px;
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            margin: 0 3px;
            background: var(--pink-muted);
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }

        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0%, 60%, 100% {
                transform: translateY(0);
            }
            30% {
                transform: translateY(-8px);
            }
        }

        /* Input Area - Fixed at bottom */
        .chat-input-area {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 20px;
            background: linear-gradient(to top, var(--bg-dark) 80%, transparent);
            z-index: 100;
        }

        .chat-input-wrapper {
            max-width: 600px;
            margin: 0 auto;
        }

        .chat-input-container {
            display: flex;
            gap: 10px;
            background: var(--bar-color);
            padding: 8px;
            border-radius: 30px;
            border: 1px solid rgba(88, 0, 99, 0.5);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
        }

        #userInput {
            flex: 1;
            padding: 12px 18px;
            border: none;
            border-radius: 25px;
            font-size: 0.95rem;
            outline: none;
            background: rgba(0, 0, 0, 0.5);
            color: var(--text-white);
            min-width: 200px;
        }

        #userInput::placeholder {
            color: var(--text-gray);
        }

        #userInput:focus {
            background: rgba(0, 0, 0, 0.7);
        }

        #sendBtn {
            padding: 12px 24px;
            background: linear-gradient(45deg, #EA2BAE 0%, #720ABC 50%, #22044D 100%);
            color: var(--text-white);
            border: none;
            border-radius: 25px;
            font-size: 0.95rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            white-space: nowrap;
        }

        #sendBtn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px var(--shadow);
        }

        #sendBtn:active {
            transform: translateY(0);
        }

        #sendBtn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        /* Scrollbar */
        .chat-messages::-webkit-scrollbar {
            width: 8px;
        }

        .chat-messages::-webkit-scrollbar-track {
            background: var(--bg-dark);
        }

        .chat-messages::-webkit-scrollbar-thumb {
            background: var(--purple-mid);
            border-radius: 10px;
        }

        /* Link styling */
        .message-bubble a {
            color: var(--pink-bright);
            text-decoration: underline;
        }

        .message-bubble a:hover {
            color: var(--pink-muted);
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .chat-header {
                padding: 12px 15px;
            }

            .logo-container {
                width: 35px;
                height: 35px;
            }

            .header-text h1 {
                font-size: 1.1rem;
            }

            .chat-messages {
                padding: 70px 15px 110px 15px;
            }

            .message-content {
                max-width: 90%;
            }

            .message-bubble {
                font-size: 0.9rem;
            }

            .message-icon {
                width: 30px;
                height: 30px;
                font-size: 1rem;
            }

            .chat-input-area {
                padding: 15px;
            }

            #sendBtn {
                padding: 10px 18px;
                font-size: 0.9rem;
            }
        }

        @media (max-width: 480px) {
            .header-text h1 {
                font-size: 1rem;
            }

            .chat-messages {
                padding: 65px 10px 100px 10px;
            }

            .message-content {
                max-width: 95%;
            }

            #userInput {
                font-size: 0.9rem;
                padding: 10px 15px;
            }
        }

        @media (max-width: 300px) {
            .logo-container {
                width: 30px;
                height: 30px;
            }

            .header-text h1 {
                font-size: 0.9rem;
            }

            .message-icon {
                width: 25px;
                height: 25px;
            }

            .message-bubble {
                font-size: 0.85rem;
                padding: 10px 14px;
            }

            #sendBtn {
                padding: 8px 12px;
                font-size: 0.85rem;
            }
        }

        @media (max-height: 500px) {
            .chat-header {
                padding: 8px 15px;
            }

            .chat-messages {
                padding: 55px 15px 90px 15px;
                min-height: 300px;
            }

            .chat-input-area {
                padding: 10px 15px;
            }
        }
    </style>
</head>
<body>
    <!-- Fixed Header -->
    <div class="chat-header">
        <div class="logo-container">
            <img src="static/yardstick.png" alt="Yardstick Logo">
        </div>
        <div class="header-text">
            <h1>Yardstick</h1>
        </div>
    </div>

    <!-- Full Screen Messages Area -->
    <div class="chat-messages" id="chatMessages">
        <div class="messages-container" id="messagesContainer">
            <!-- Welcome Message -->
            <div class="message bot">
                <div class="message-content">
                    <div class="message-icon"><img src="static/yardstick.png" alt="Yardstick Logo"></div>
                    <div class="message-bubble">
                        Hello! Welcome to Yardstick. How can I assist you today?
                    </div>
                </div>
            </div>
        </div>

        <!-- Typing Indicator -->
        <div class="typing-indicator" id="typingIndicator">
            <div class="typing-indicator-icon"></div>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    </div>

    <!-- Fixed Bottom Input Area -->
    <div class="chat-input-area">
        <div class="chat-input-wrapper">
            <div class="chat-input-container">
                <input 
                    type="text" 
                    id="userInput" 
                    placeholder="Ask anything about Yardstick's AI services"
                    autocomplete="off"
                >
                <button id="sendBtn">Send</button>
            </div>
        </div>
    </div>

    <script>
        const chatMessages = document.getElementById('chatMessages');
        const messagesContainer = document.getElementById('messagesContainer');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');
        const typingIndicator = document.getElementById('typingIndicator');

        // Add message to chat
        function addMessage(text, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            const icon = document.createElement('div');
            icon.className = 'message-icon';
            if (isUser) {
                icon.textContent = '👤';
            } else {
                const logoImg = document.createElement('img');
                logoImg.src = 'static/yardstick.png';
                logoImg.alt = 'Yardstick';
                icon.appendChild(logoImg);
            }
            
            const bubble = document.createElement('div');
            bubble.className = 'message-bubble';
            
            // Convert URLs to clickable links
            const urlRegex = /(https?:\/\/[^\s]+)/g;
            const linkedText = text.replace(urlRegex, (url) => {
                return `<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`;
            });
            
            bubble.innerHTML = linkedText;
            
            contentDiv.appendChild(icon);
            contentDiv.appendChild(bubble);
            messageDiv.appendChild(contentDiv);
            
            messagesContainer.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Show/hide typing indicator
        function showTyping(show) {
            typingIndicator.classList.toggle('active', show);
            if (show) {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }

        // Send message
        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;

            // Add user message
            addMessage(message, true);
            userInput.value = '';
            sendBtn.disabled = true;

            // Show typing
            showTyping(true);

            try {
                // Call your backend API here
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question: message })
                });

                const data = await response.json();

                // Hide typing and show response
                showTyping(false);
                addMessage(data.answer, false);

            } catch (error) {
                showTyping(false);
                addMessage("Sorry, I'm having trouble connecting. Please try again.", false);
            }

            sendBtn.disabled = false;
            userInput.focus();
        }

        // Event listeners
        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Focus input on load
        userInput.focus();
    </script>

</body>
</html>


    '''
    return render_template_string(html_content)

@app.route('/api/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid request'}), 400
        
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        print(f"📥 Question: {question}")
        answer = generate_answer(question)
        print(f"📤 Answer: {answer[:100]}...")
        
        return jsonify({'answer': answer}), 200
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Server error. Please try again.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

