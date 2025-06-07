// src/api/api.ts
export async function fetchDocuments(): Promise<string[]> {
  const res = await fetch("http://localhost:8000/api/documents");
  if (!res.ok) throw new Error("Failed to fetch documents");
  const data = await res.json();
  return data.documents;
}
