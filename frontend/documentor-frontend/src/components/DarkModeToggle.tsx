// src/components/DarkModeToggle.tsx
import React, { useState, useEffect } from "react";

const DarkModeToggle: React.FC = () => {
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem("darkMode") === "true";
  });

  useEffect(() => {
    if (darkMode) {
      document.body.classList.add("dark-mode");
    } else {
      document.body.classList.remove("dark-mode");
    }
    localStorage.setItem("darkMode", darkMode.toString());
  }, [darkMode]);

  return (
    <button
      onClick={() => setDarkMode((prev) => !prev)}
      style={{
        padding: "0.5rem 1rem",
        borderRadius: "6px",
        backgroundColor: darkMode ? "#f9f9f9" : "#333",
        color: darkMode ? "#000" : "#fff",
        border: "none",
        cursor: "pointer",
      }}
    >
      {darkMode ? "☀ Light Mode" : "🌙 Dark Mode"}
    </button>
  );
};

export default DarkModeToggle;
