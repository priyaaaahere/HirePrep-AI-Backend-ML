"""
Test script to verify all ML models are working
"""

print("="*60)
print("TESTING ML MODELS")
print("="*60)

# Test 1: Placement Predictor
print("\n1. Testing Placement Predictor...")
try:
    from app.ml.placement_predictor import predict_base_probability
    
    test_profile = {
        "education": {
            "cgpa": 8.5
        },
        "projects": [{"name": "Project 1"}, {"name": "Project 2"}, {"name": "Project 3"}],
        "certifications": [{"name": "Cert 1"}, {"name": "Cert 2"}],
        "internships": [{"company": "Tech Corp"}],
        "skills": ["Python", "Java", "SQL", "Git", "React"],
        "placement_inputs": {
            "communication_rating": 4.0,
            "hackathon": "Yes",
            "twelfth_percent": 85.0,
            "tenth_percent": 90.0,
            "backlogs": 0
        }
    }
    
    probability = predict_base_probability(test_profile)
    status = "Likely Placed" if probability >= 50 else "May Need Improvement"
    print(f"   ✓ Placement Probability: {probability:.2f}%")
    print(f"   ✓ Status: {status}")
    print(f"   Status: WORKING")
except Exception as e:
    print(f"   ✗ Error: {e}")
    print(f"   Status: FAILED")

# Test 2: Role Recommender
print("\n2. Testing Role Recommender...")
try:
    from app.ml.role_recommender import recommend_roles
    
    test_skills = ["Python", "Machine Learning", "TensorFlow", "Data Analysis"]
    
    recommendations = recommend_roles(
        resume_skills=test_skills,
        experience_years=2,
        top_n=5
    )
    
    print(f"   ✓ Found {len(recommendations)} role recommendations")
    if recommendations:
        print(f"   ✓ Top role: {recommendations[0]['role']}")
        print(f"   ✓ Match score: {recommendations[0]['match_score']:.2f}%")
    print(f"   Status: WORKING")
except Exception as e:
    print(f"   ✗ Error: {e}")
    print(f"   Status: FAILED")

# Test 3: Skill Matcher
print("\n3. Testing Skill Matcher...")
try:
    from app.ml.skill_matcher import compute_skill_match
    
    test_resume_skills = ["Python", "Java", "SQL", "Git"]
    
    result = compute_skill_match(
        resume_skills=test_resume_skills,
        desired_role="Software Developer",
        experience_level="Entry"
    )
    
    print(f"   ✓ Match percentage: {result['skill_match_percent']:.2f}%")
    print(f"   ✓ Matched skills: {len(result['matched_skills'])}")
    print(f"   ✓ Missing skills: {len(result['missing_skills'])}")
    print(f"   Status: WORKING")
except Exception as e:
    print(f"   ✗ Error: {e}")
    print(f"   Status: FAILED")

print("\n" + "="*60)
print("MODEL TESTING COMPLETE")
print("="*60)
