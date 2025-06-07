// UploadForm.tsx
import React, { useState } from "react";

const UploadForm: React.FC = () => {
  const [files, setFiles] = useState<FileList | null>(null);
  const [status, setStatus] = useState("");

  const handleUpload = async () => {
    if (!files) return;

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }

    try {
      const res = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setStatus(data.message);
    } catch (err) {
      setStatus("Upload failed.");
    }
  };

  return (
    <div>
      <h2>Upload Documents</h2>
      <input type="file" multiple onChange={(e) => setFiles(e.target.files)} />
      <button onClick={handleUpload}>Upload</button>
      {status && <p>{status}</p>}
    </div>
  );
};

export default UploadForm;