


User Data Inputs

Users have two types of data sources: MSA (with 5 records) and NDA (with 10 records). The user can request to send either MSA data or NDA data (or both) to the respective target databases (Prod A for MSA data, Prod B for NDA data).
Safety and Intent Detection (LLMs)

A first LLM (called “Safety LLM” or “Llama Guard”) oversees the initial processing step.
It routes the request to an Intent Detection LLM, which figures out whether the user wants to send all records, only MSA records, or only NDA records.
Extraction & Validation

Another LLM (“Extract the Information LLM”) pulls out the specific information that needs to be forwarded.
A separate LLM (“Judge LLM”) then evaluates whether the extracted data is acceptable and aligns with the user’s stated intent. This step checks for any issues like PHI (protected health information) or other sensitive data that should not be shared if it does not match user intent.
Human‐in‐the‐Loop vs. Automatic Approval

After the data is classified and extracted, the system determines whether human confirmation is required (for instance, due to sensitive content or policy settings).
If no confirmation is required, the data goes directly to the production databases (Prod A for MSA, Prod B for NDA).
If confirmation is required, the data first appears in a staging dashboard. A human user can review the records, possibly edit any flagged items, and then approve or reject them.
Staging Dashboard

The dashboard allows reviewers to see which records are pending approval.
They can modify records that the system has flagged as “non‐good” (e.g., incomplete, inaccurate, sensitive).
Once approved, those records are then sent to the respective production database.
Embedding / Vector DB

In the background, there is also a mention of an embedding process to a vector database, likely for later retrieval or semantic search. This piece ensures that the data (particularly MSA or NDA records) is stored in a way that is easy to retrieve or further processed with LLMs.
