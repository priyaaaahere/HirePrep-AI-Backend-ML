import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a resume PDF file.
    """
    pages_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

    return "\n".join(pages_text)
