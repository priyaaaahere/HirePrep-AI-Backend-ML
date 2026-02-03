import re
from app.pdf_to_text import extract_text_from_pdf
from app.resume_parser import clean_text, extract_section, is_project_header

text = extract_text_from_pdf('data/resume.pdf')
cleaned = clean_text(text)
projects_section = extract_section(cleaned, 'PROJECTS')

# Test first line
lines = projects_section.split('\n')
line0 = lines[0].strip()
print('Line 0:', repr(line0))
print('is_project_header:', is_project_header(line0))

# Test pattern manually  
pattern = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[''\" ]?\d{2,4}\s*[-–—]\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Present|Current|Ongoing)"
match = re.search(pattern, line0, re.IGNORECASE)
print('Pattern match:', match)

# What's the actual character between dates?
for i, c in enumerate(line0):
    if c in ['-', '–', '—', ' ']:
        print(f'Pos {i}: char={repr(c)} ord={ord(c)}')
