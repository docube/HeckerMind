// src/components/DocumentList.tsx
import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchDocuments } from "../api/api";

const DocumentList: React.FC = () => {
  const { data, isLoading, error } = useQuery<string[]>({
    queryKey: ["documents"],
    queryFn: fetchDocuments,
  });

  const [showAll, setShowAll] = useState(false);
  const [search, setSearch] = useState("");

  const filteredDocs = (data ?? []).filter((doc) =>
    doc.toLowerCase().includes(search.toLowerCase())
  );

  const displayedDocs = showAll ? filteredDocs : filteredDocs.slice(0, 5);

  return (
    <div className="section">
      <h2 className="section-title">📁 Uploaded Documents</h2>

      <input
        type="text"
        placeholder="Search documents..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="doc-search"
      />

      {isLoading && <p>Loading...</p>}
      {error && <p className="text-red-600">Failed to load documents.</p>}

      <ul className="doc-list">
        {displayedDocs.map((doc) => (
          <li key={doc}>
            📄 {doc}
            <a
              href={`http://localhost:8000/uploads/${encodeURIComponent(doc)}`}
              target="_blank"
              rel="noopener noreferrer"
              className="download-link"
            >
              Download
            </a>
          </li>
        ))}
      </ul>

      {filteredDocs.length > 5 && (
        <button onClick={() => setShowAll((prev) => !prev)} className="toggle-btn">
          {showAll ? "Show Less" : "Show All"}
        </button>
      )}
    </div>
  );
};

export default DocumentList;
