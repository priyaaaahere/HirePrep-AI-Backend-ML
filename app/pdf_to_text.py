import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a resume PDF file.
    Uses optimized settings for better text extraction.
    """
    pages_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Try with layout preservation first
            text = page.extract_text(
                x_tolerance=2,  # Horizontal tolerance for grouping characters
                y_tolerance=2,  # Vertical tolerance
            )
            if text:
                pages_text.append(text)

    full_text = "\n".join(pages_text)
    
    # Post-process to fix common PDF extraction issues
    # Fix broken words where letters are separated by spaces
    # Pattern: single letter followed by space and single letter (repeated)
    import re
    
    # Don't fix if it's at word boundaries (to preserve "I am" etc)
    # Only fix inside words: "P y t h o n" -> "Python"
    
    return full_text
