/**
 * Fitness AI Coach - Main Frontend Logic
 * - Handles: Chat interactions, Stat Updates, Chart Rendering
 */

document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const chatStream = document.getElementById('chat-stream');
    const userInput = document.getElementById('user-input');
    const statusDot = document.querySelector('.status-dot');

    // Load Chat History from LocalStorage
    loadHistory();
    refreshDashboard();

    let isProcessing = false;

    /**
     * Serializes and saves the current chat stream to localStorage
     */
    function saveHistory() {
        const messages = [];
        const bubbles = chatStream.querySelectorAll('.msg');
        bubbles.forEach(div => {
            // Don't save temporary UI states like 'thinking'
            if (div.id && div.id.startsWith('thinking')) return;
            
            messages.push({
                text: div.querySelector('.bubble').innerText,
                role: div.classList.contains('assistant') ? 'assistant' : 'user'
            });
        });
        localStorage.setItem('synapse_chat_history', JSON.stringify(messages));
    }

    /**
     * Restores chat history from localStorage
     */
    function loadHistory() {
        const saved = localStorage.getItem('synapse_chat_history');
        if (saved) {
            const messages = JSON.parse(saved);
            if (messages.length > 0) {
                chatStream.innerHTML = ''; // Clear default greeting if we have history
                messages.forEach(m => addMessage(m.text, m.role, null, false));
            }
        }
    }

    // Handle Chat Submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (isProcessing) return;
        const prompt = userInput.value.trim();
        if (!prompt) return;

        // 1. Add User Message to UI
        addMessage(prompt, 'user');
        userInput.value = '';
        setLoading(true);

        // 2. Add temporary "Thinking" bubble
        const thinkingId = 'thinking-' + Date.now();
        addMessage("Coach is thinking...", 'assistant', thinkingId);

        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt })
            });

            if (!response.ok) throw new Error(`Server responded with ${response.status}`);

            const data = await response.json();

            // Remove thinking bubble and add real answer
            document.getElementById(thinkingId)?.remove();

            if (data.answer) {
                addMessage(data.answer, 'assistant');
                refreshDashboard();
            } else {
                addMessage("Brain is a bit foggy. Try rephrasing that?", 'assistant');
            }
        } catch (err) {
            console.error("Chat Error:", err);
            document.getElementById(thinkingId)?.remove();
            addMessage(`⚠️ Connection failed: ${err.message}. Is the Flask server alive?`, 'assistant');
        } finally {
            setLoading(false);
        }
    });

    /**
     * Appends a message bubble to the chat stream
     */
    function addMessage(text, role, id = null, shouldSave = true) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `msg ${role}`;
        if (id) msgDiv.id = id;
        
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerText = role === 'assistant' ? '🤖' : '💪';

        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.innerText = text;

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(bubble);
        chatStream.appendChild(msgDiv);
        
        // Auto-scroll to bottom
        setTimeout(() => {
            chatStream.scrollTop = chatStream.scrollHeight;
        }, 50);

        // Save to local storage after adding
        if (shouldSave) saveHistory();
    }

    /**
     * Updates the sidebar with latest stats from the backend
     */
    async function refreshDashboard() {
        try {
            const res = await fetch('/api/stats');
            const stats = await res.json();
            
            document.getElementById('user-name').innerText = stats.name || "Athlete";
            document.getElementById('user-weight').innerText = `${stats.weight} kg`;
            document.getElementById('user-bf').innerText = `${stats.bf_pct} %`;
            document.getElementById('user-goal').innerText = stats.goal;
            document.getElementById('user-split').innerText = stats.split;
            document.getElementById('user-bench').innerText = stats.bench_pr;
        } catch (err) {
            console.error("Dashboard refresh failed:", err);
        }
    }

    /**
     * Visual feedback for the 'Thinking' state
     */
    function setLoading(isLoading) {
        isProcessing = isLoading;
        const submitBtn = chatForm.querySelector('button');
        if (isLoading) {
            statusDot.style.backgroundColor = '#fbbf24'; // Warning color (Yellow)
            statusDot.style.boxShadow = '0 0 10px #fbbf24';
            if (submitBtn) submitBtn.disabled = true;
        } else {
            statusDot.style.backgroundColor = '#a3e635'; // Ready color (Neon Green)
            statusDot.style.boxShadow = '0 0 10px #a3e635';
            if (submitBtn) submitBtn.disabled = false;
        }
    }
});
