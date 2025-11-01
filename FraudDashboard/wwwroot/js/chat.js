// =========================
// AI Analyst Chat Client
// =========================

const AI_API_URL = "http://localhost:8080/query";

const chatWidget = document.getElementById("chatWidget");
const chatToggle = document.getElementById("chatToggle");
const chatCollapseBtn = document.getElementById("chatCollapseBtn");
const chatBody = document.getElementById("chatBody");
const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const chatText = document.getElementById("chatText");

// ---- UI helpers ----
function appendBubble(text, who) {
    const bubble = document.createElement("div");
    bubble.classList.add("chat-bubble");

    if (who === "user") {
        bubble.classList.add("from-user");
    } else if (who === "ai") {
        bubble.classList.add("from-ai");
    } else if (who === "thinking") {
        bubble.classList.add("from-ai", "thinking");
    }

    bubble.textContent = text;
    chatMessages.appendChild(bubble);

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendUserChat(text) {
    appendBubble(text, "user");
}

function appendAIChat(text) {
    appendBubble(text, "ai");
}

let thinkingBubbleEl = null;
function showThinking() {
    thinkingBubbleEl = document.createElement("div");
    thinkingBubbleEl.classList.add("chat-bubble", "from-ai", "thinking");
    thinkingBubbleEl.textContent = "Analyzing your question...";
    chatMessages.appendChild(thinkingBubbleEl);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideThinking() {
    if (thinkingBubbleEl && thinkingBubbleEl.parentNode) {
        thinkingBubbleEl.parentNode.removeChild(thinkingBubbleEl);
    }
    thinkingBubbleEl = null;
}

// ---- backend call ----
async function callAiBackend(questionText) {
    const q = questionText.trim();
    if (!q) {
        return { error: "Empty question" };
    }

    try {
        const resp = await fetch(AI_API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: q,
                summarize: true
            })
        });

        if (!resp.ok) {
            const detail = await resp.text();
            return { error: `Backend error (${resp.status}): ${detail}` };
        }

        const data = await resp.json();
        // data = { sql, rows, summary }

        let finalText = "";

        if (data.summary) {
            finalText += data.summary.trim() + "\n\n";
        }

        if (data.rows && data.rows.length > 0) {
            finalText += "Sample results:\n";
            data.rows.slice(0, 5).forEach(rowObj => {
                finalText += "• " + JSON.stringify(rowObj) + "\n";
            });
            finalText += "\n";
        }

        if (data.sql) {
            finalText += "SQL executed:\n" + data.sql;
        }

        if (!finalText.trim()) {
            finalText = "(No answer returned)";
        }

        return { ok: true, text: finalText };

    } catch (err) {
        return { error: "Request failed: " + err.message };
    }
}

// ---- send flow ----
async function sendChat(questionText) {
    // open widget if closed
    chatWidget.classList.add("open");
    chatBody.style.display = "flex";

    showThinking();
    const result = await callAiBackend(questionText);
    hideThinking();

    if (result.error) {
        appendAIChat("⚠ " + result.error);
    } else {
        appendAIChat(result.text);
    }
}

// ---- hook UI events ----
if (chatToggle) {
    chatToggle.addEventListener("click", () => {
        const isOpen = chatWidget.classList.contains("open");
        if (isOpen) {
            chatWidget.classList.remove("open");
            chatBody.style.display = "none";
            if (chatCollapseBtn) chatCollapseBtn.textContent = "▲";
        } else {
            chatWidget.classList.add("open");
            chatBody.style.display = "flex";
            if (chatCollapseBtn) chatCollapseBtn.textContent = "▼";
        }
    });
}

if (chatForm) {
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = chatText.value.trim();
        if (!text) return;

        appendUserChat(text);
        await sendChat(text);
        chatText.value = "";
    });
}

// expose for other scripts (like askAiWhy in modal)
window.appendUserChat = appendUserChat;
window.sendChat = sendChat;
