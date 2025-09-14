import spacy
from docx import Document
import os
import json

nlp = spacy.load("en_core_web_sm")

def extract_clean_text(docx_path):
    """Extract and clean text from a .docx memo."""
    docx = Document(docx_path)
    text = "\n".join([para.text for para in docx.paragraphs])
    doc = nlp(text)
    
    domain_stopwords = {
        "student", "students", "school", "memo", "memorandum",
        "office", "university", "college", "campus", "department",
        "faculty", "member", "personnel", "program", "psu", "lingayen"
    }
    
    clean_tokens = [
        token.lemma_.lower().strip()
        for token in doc
        if token.lemma_ not in ["-PRON-"]      # remove pronouns
        and not token.is_stop                  # remove stopwords
        and not token.is_punct                 # remove punctuation
        and not token.like_num                 # remove numbers
        and token.is_alpha                     # only words
        and token.lemma_.lower() not in domain_stopwords
        and token.ent_type_ not in ["DATE", "TIME", "PERSON", "ORG", "GPE", "FAC", "NORP"]
        and token.pos_ in {"NOUN", "ADJ"}      # keep only nouns/adjectives
    ]
    
    clean_text = " ".join(clean_tokens)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    return {
        "filename": os.path.basename(docx_path),
        "raw_text": text,
        "clean_text": clean_text,
        "tokens": clean_tokens,
        "sentences": sentences,
    }

def process_memos(folder_path):
    """Process all .docx memos in a folder."""
    results = []
    for file in os.listdir(folder_path):
        if file.endswith(".docx"):
            file_path = os.path.join(folder_path, file)
            memo_data = extract_clean_text(file_path)
            results.append(memo_data)
    return results


folder = "./memos"
memos = process_memos(folder)

# print(f"Processed {len(memos)} memos")
# print("Sample memo keys:", memos[0].keys())
# print("First memo clean tokens sample:", memos[0]["tokens"][:20])
# print(json.dumps(memos, indent=2, ensure_ascii=False))
with open("memos.json", "w", encoding="utf-8") as f:
    json.dump(memos, f, indent=2, ensure_ascii=False)
