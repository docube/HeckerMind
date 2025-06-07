// src/components/ChatBox.tsx
import React, { useState, useEffect, useRef } from "react";

interface Message {
  id: number;
  sender: "user" | "bot";
  text: string;
  timestamp: string;
  sources?: string[]; // For matched chunks
  suggestions?: string[]; // Follow-up questions
}

const ChatBox: React.FC = () => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const messageEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg: Message = {
      id: Date.now(),
      sender: "user",
      text: input,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMsg.text }),
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let botMessage = "";

      const messageId = Date.now() + 1;

      // Add placeholder
      setMessages((prev) => [
        ...prev,
        {
          id: messageId,
          sender: "bot",
          text: "",
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        botMessage += decoder.decode(value, { stream: true });

        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId ? { ...msg, text: botMessage } : msg
          )
        );
      }

      // Simulate highlighting & suggestions
      const simulatedChunks = [
        "• Company allows bonuses based on project performance.",
        "• Severance is paid out over 3 months post-departure.",
      ];

      const followUps = [
        "What is the severance package for managers?",
        "How often are bonuses evaluated?",
        "Can bonuses be withdrawn?"
      ];

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === messageId
            ? { ...msg, sources: simulatedChunks, suggestions: followUps }
            : msg
        )
      );
    } catch (err) {
      console.error("Error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
  };

  return (
    <div className="section">
      <h2 className="section-title">💬 Chat with Documents</h2>

      <div className="chat-window">
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-bubble ${msg.sender}`}>
            <div className="message-text">
              {msg.sender === "user" ? "🧑‍💼" : "🤖"} {msg.text}
            </div>
            <div className="timestamp">{msg.timestamp}</div>

            {/* Display matched chunks */}
            {msg.sources && msg.sources.length > 0 && (
              <div className="matched-chunks">
                <strong>📚 Matched Chunks:</strong>
                <ul>
                  {msg.sources.map((src, idx) => (
                    <li key={idx}>- {src}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Display suggestions */}
            {msg.suggestions && msg.suggestions.length > 0 && (
              <div className="suggestions">
                <strong>💡 Follow-up Questions:</strong>
                <div className="suggestion-list">
                  {msg.suggestions.map((suggestion, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleSuggestionClick(suggestion)}
                      className="suggestion-btn"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="chat-bubble bot">
            <div className="message-text">🤖 Thinking...</div>
          </div>
        )}

        <div ref={messageEndRef}></div>
      </div>

      <div className="chat-input">
        <input
          type="text"
          placeholder="Ask a question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
        />
        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
};

export default ChatBox;
