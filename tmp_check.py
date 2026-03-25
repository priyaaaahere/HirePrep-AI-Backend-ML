import asyncio
from app.main import complete_analysis

profile = {
    "education": { "cgpa": "8.42" },
    "skills": ["Java", "Python", "Git", "GitHub", "MySQL", "Microsoft Excel", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Tkinter", "Scikit-learn", "TensorFlow", "StatsModels", "Teamwork", "Curious", "Flexible", "Growth Mindset"," Data Structures and Algorithm", "Data Visualization", "Machine Learning","Keras", "Data preprocessing","Feature Engineering","EDA"],
    "projects": [{}, {}, {}],
    "certifications": ["c1", "c2","c3"],
    "internships": [],
    "experience": { "years": 0 },
    "career_preferences": {
        "desired_role": "Data Science Associate"
    },
    "placement_inputs": {
        "communication_rating": 4.2,
        "hackathon": "No",
        "twelfth_percent": 90.4,
        "tenth_percent": 96.4,
        "backlogs": 0
    }
}

result = asyncio.run(complete_analysis(profile))
print(result.get('status'))
print(result.get('role_recommendations',{}).get('source'))
print(result.get('skill_analysis',{}).get('missing_skills'))
print(result.get('role_recommendations',{}).get('top_roles',[{}])[0].get('role'))
