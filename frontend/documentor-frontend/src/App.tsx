// src/App.tsx
import React from "react";
import UploadForm from "./components/UploadForm";
import DocumentList from "./components/DocumentList";
import ChatBox from "./components/ChatBox";
import DarkModeToggle from "./components/DarkModeToggle";
import logo from "./assets/DocuMentor_Logo.png";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

// 🧠 Setup query client
const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="app">
        <header className="header">
          <img src={logo} alt="DocuMentor Logo" className="logo" />
          <h1>HeckerMind</h1>
          <DarkModeToggle />
        </header>

        <main className="container">
          <UploadForm />
          <DocumentList />
          <ChatBox />
        </main>
      </div>
    </QueryClientProvider>
  );
}

export default App;
