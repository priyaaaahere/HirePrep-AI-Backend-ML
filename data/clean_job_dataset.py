"""
Dataset Cleaning Script for job_dataset.csv - Version 2
=========================================================
This script cleans and preprocesses the job dataset for ML models and analysis.

Issues addressed:
1. Remove unused columns (JobID, Responsibilities, Keywords)
2. Clean Title column - Remove experience suffixes (- Fresher, - Experienced, etc.)
3. Standardize role names comprehensively (all variations → canonical names)
4. Handle YearsOfExperience ranges - Convert to numeric (min value)
5. Aggregate skills for duplicate titles per experience level

Output:
- job_dataset_cleaned.csv: Cleaned dataset with original structure
- job_dataset_aggregated.csv: Aggregated dataset with unique Title + ExperienceLevel combinations
"""

import pandas as pd
import re
import os

# File paths
INPUT_PATH = "data/job_dataset.csv"
OUTPUT_CLEANED_PATH = "data/job_dataset_cleaned.csv"
OUTPUT_AGGREGATED_PATH = "data/job_dataset_aggregated.csv"


def load_data():
    """Load the original dataset"""
    df = pd.read_csv(INPUT_PATH)
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    return df


def remove_unused_columns(df):
    """Remove columns not used for analysis"""
    columns_to_remove = ['JobID', 'Responsibilities', 'Keywords']
    existing_columns = [col for col in columns_to_remove if col in df.columns]
    df = df.drop(columns=existing_columns)
    print(f"\nRemoved columns: {existing_columns}")
    print(f"Remaining columns: {list(df.columns)}")
    return df


def clean_title(title):
    """
    Remove experience-related suffixes from job titles.
    """
    if pd.isna(title):
        return title
    
    # Patterns to remove (case insensitive)
    patterns_to_remove = [
        r'\s*-\s*Fresher\s*$',
        r'\s*-\s*Experienced\s*$',
        r'\s*-\s*Entry[- ]?Level\s*$',
        r'\s*-\s*Mid[- ]?Level\s*$',
        r'\s*-\s*Senior[- ]?Level\s*$',
        r'\s*-\s*Entry Level\s*$',
        r'\s*-\s*Mid Level\s*$',
        r'\s*-\s*Senior Level\s*$',
        r'\s*-\s*Junior\s*$',
        r'\s*-\s*Intern\s*$',
        r'\s*-\s*Trainee\s*$',
    ]
    
    cleaned = title
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip()


# ============================================================
# COMPREHENSIVE TITLE STANDARDIZATION MAPPING
# ============================================================
# This maps all role variations to canonical standardized names

TITLE_MAPPING = {
    # =====================================================
    # MACHINE LEARNING ENGINEER - All variations
    # =====================================================
    'ML Engineer': 'Machine Learning Engineer',
    'ML Engineer Intern': 'Machine Learning Engineer',
    'Fresher ML Engineer': 'Machine Learning Engineer',
    'Entry-Level ML Engineer': 'Machine Learning Engineer',
    'ML Engineer Trainee': 'Machine Learning Engineer',
    'Junior ML Engineer': 'Machine Learning Engineer',
    'Junior Machine Learning Engineer': 'Machine Learning Engineer',
    'Senior ML Engineer': 'Machine Learning Engineer',
    'Lead ML Engineer': 'Machine Learning Engineer',
    'Principal ML Engineer': 'Machine Learning Engineer',
    'Senior ML Infrastructure Engineer': 'Machine Learning Engineer',
    'Lead ML/AI Engineer': 'AI/ML Engineer',
    'Lead Machine Learning Engineer': 'Machine Learning Engineer',
    'Senior Machine Learning Engineer': 'Machine Learning Engineer',
    'Principal Machine Learning Engineer': 'Machine Learning Engineer',
    
    # =====================================================
    # DATA SCIENTIST - All variations
    # =====================================================
    'Data Scientist - Entry Level': 'Data Scientist',
    'Junior Data Scientist': 'Data Scientist',
    'Data Analyst / Data Scientist Intern': 'Data Scientist',
    'Data Scientist - Fresher': 'Data Scientist',
    'Fresher Data Scientist': 'Data Scientist',
    'Data Science Associate': 'Data Science Associate',
    'Data Science Intern': 'Data Scientist',
    'Data Science Team Lead': 'Data Scientist',
    'Machine Learning Scientist': 'Machine Learning Scientist',
    'Lead Machine Learning Scientist': 'Machine Learning Scientist',
    'Entry Level Data Scientist': 'Data Scientist',
    'Lead Data Scientist': 'Data Scientist',
    'Senior Data Scientist': 'Data Scientist',
    'Principal Data Scientist': 'Data Scientist',
    
    # =====================================================
    # DATA ANALYST - All variations
    # =====================================================
    'Data Analyst - Fresher': 'Data Analyst',
    'Data Analyst - Experienced': 'Data Analyst',
    
    # =====================================================
    # CLOUD ENGINEER - All variations (consolidate all cloud roles)
    # =====================================================
    'Cloud Engineer - Fresher': 'Cloud Engineer',
    'Cloud Engineer - Experienced': 'Cloud Engineer',
    'Cloud Architect': 'Cloud Architect',
    'Cloud Solutions Associate': 'Cloud Solutions Associate',
    'Junior Cloud Engineer': 'Cloud Engineer',
    'Cloud Trainee': 'Cloud Engineer',
    'Associate Cloud Architect': 'Associate Cloud Architect',
    'Senior Cloud Architect': 'Cloud Architect',
    'Lead Cloud Solutions Architect': 'Cloud Solutions Architect',
    'Cloud Security Architect': 'Cloud Security Architect',
    'DevOps Cloud Architect': 'DevOps Cloud Architect',
    'Cloud Infrastructure Architect': 'Cloud Architect',
    'Multi-Cloud Architect': 'Multi-Cloud Architect',
    'Cloud Migration Specialist': 'Cloud Migration Specialist',
    'Cloud Automation Engineer': 'Cloud Automation Engineer',
    'Cloud Platform Architect': 'Cloud Architect',
    'Cloud DevSecOps Architect': 'Cloud Architect',
    'Enterprise Cloud Architect': 'Cloud Architect',
    'Hybrid Cloud Architect': 'Hybrid Cloud Architect',
    'Cloud Cost Optimization Architect': 'Cloud Cost Optimization Architect',
    'Cloud Infrastructure Security Architect': 'Cloud Infrastructure Security Architect',
    'Cloud Network Engineer': 'Cloud Network Engineer',
    'Cloud Security Analyst': 'Cloud Security Analyst',
    
    # =====================================================
    # DEVOPS ENGINEER - All variations
    # =====================================================
    'DevOps Engineer - Fresher': 'DevOps Engineer',
    'DevOps Engineer - Experienced': 'DevOps Engineer',
    
    # =====================================================
    # ANDROID DEVELOPER - All variations
    # =====================================================
    'Android Developer Intern': 'Android Developer',
    'Junior Android Developer': 'Android Developer',
    'Android App Developer Trainee': 'Android App Developer',
    'Entry-level Android Developer': 'Android Developer',
    'Associate Android Developer': 'Android Developer',
    'Trainee Android App Developer': 'Android App Developer',
    'Junior Android Engineer': 'Android Engineer',
    'Android Development Intern': 'Android Developer',
    'Android Developer Trainee': 'Android Developer',
    'Senior Android Developer': 'Android Developer',
    'Lead Android Engineer': 'Android Engineer',
    'Android Solutions Engineer': 'Android Solutions Engineer',
    'Principal Android Developer': 'Android Developer',
    'Android Tech Lead': 'Android Developer',
    'Senior Android Engineer': 'Android Engineer',
    'Android Architect': 'Android Architect',
    'Android Solutions Architect': 'Android Solutions Architect',
    'Principal Android Engineer': 'Android Engineer',
    'Android App Developer': 'Android Developer',
    'Android Engineer': 'Android Engineer',
    
    # =====================================================
    # iOS DEVELOPER - All variations
    # =====================================================
    'iOS App Developer Intern': 'iOS App Developer',
    'iOS Application Developer': 'iOS App Developer',
    'iOS Architect': 'iOS Architect',
    'iOS Developer Trainee': 'iOS Developer',
    'iOS Junior Developer': 'iOS Developer',
    'iOS Mobile App Engineer': 'iOS Mobile App Engineer',
    'iOS Mobile Developer': 'iOS Mobile Developer',
    'iOS Solutions Engineer': 'iOS Solutions Engineer',
    'iOS Tech Lead': 'iOS Developer',
    'Junior iOS Developer': 'iOS Developer',
    'Junior Swift Developer': 'Swift Developer',
    'Senior iOS Developer': 'iOS Developer',
    'Senior iOS Engineer': 'iOS Engineer',
    'Principal iOS Developer': 'iOS Developer',
    'Lead iOS Engineer': 'iOS Engineer',
    'Associate iOS Engineer': 'iOS Engineer',
    'Entry-level iOS Developer': 'iOS Developer',
    'Graduate iOS Developer': 'iOS Developer',
    'Trainee iOS Developer': 'iOS Developer',
    
    # =====================================================
    # NETWORK ENGINEER - All variations
    # =====================================================
    'Junior Network Engineer': 'Network Engineer',
    'Network Support Engineer': 'Network Support Engineer',
    'Associate Network Engineer': 'Network Engineer',
    'Trainee Network Engineer': 'Network Engineer',
    'Network Operations Center (NOC) Engineer': 'Network Engineer',
    'Junior Security & Network Engineer': 'Security & Network Engineer',
    'Graduate Network Engineer': 'Network Engineer',
    'Entry-Level Network Analyst': 'Network Analyst',
    'Network Intern': 'Network Engineer',
    'IT Support - Networking Focus': 'Network Engineer',
    'Senior Network Engineer': 'Network Engineer',
    'Network Security Engineer': 'Network Security Engineer',
    'Lead Network Engineer': 'Network Engineer',
    'Infrastructure Network Engineer': 'Network Engineer',
    'Network Automation Engineer': 'Network Automation Engineer',
    'Enterprise Network Engineer': 'Network Engineer',
    'Principal Network Engineer': 'Network Engineer',
    'Telecom & Network Engineer': 'Network Engineer',
    'Network Analyst': 'Network Analyst',
    
    # =====================================================
    # CYBERSECURITY ANALYST - All variations
    # =====================================================
    'Junior Security Analyst': 'Security Analyst',
    'Cybersecurity Intern': 'Cybersecurity Analyst',
    'Associate Cybersecurity Analyst': 'Cybersecurity Analyst',
    'Entry-Level Security Analyst': 'Security Analyst',
    'Trainee Security Analyst': 'Security Analyst',
    'Cybersecurity Support Analyst': 'Cybersecurity Support Analyst',
    'Graduate Security Analyst': 'Cybersecurity Analyst',
    'Cybersecurity Trainee': 'Cybersecurity Analyst',
    'Information Security Analyst - Fresher': 'Cybersecurity Analyst',
    'Senior Security Analyst': 'Cybersecurity Analyst',
    'SOC Analyst': 'Cybersecurity Analyst',
    'Lead Cybersecurity Analyst': 'Cybersecurity Analyst',
    'Cybersecurity Risk Analyst': 'Cybersecurity Risk Analyst',
    'Information Security Specialist': 'Information Security Analyst',
    'Cyber Defense Analyst': 'Cyber Defense Analyst',
    'Senior Incident Response Analyst': 'Incident Response Analyst',
    'Principal Security Analyst': 'Cybersecurity Analyst',
    'Information Security Analyst': 'Information Security Analyst',
    'Cybersecurity': 'Cybersecurity Analyst',
    
    # =====================================================
    # SYSTEM ENGINEER - All variations
    # =====================================================
    'Junior System Engineer': 'System Engineer',
    'System Support Engineer': 'System Support Engineer',
    'IT Infrastructure Engineer': 'IT Infrastructure Engineer',
    'System Engineer - Cloud & Automation': 'System Engineer',
    'Lead System Engineer': 'System Engineer',
    'Associate System Engineer': 'System Engineer',
    'System Engineer Intern': 'System Engineer',
    'System Engineer Trainee': 'System Engineer',
    'Senior System Engineer': 'System Engineer',
    'System Engineer - Automation Specialist': 'Automation Specialist',
    'System Engineer - Cloud Focus': 'System Engineer',
    'System Engineer - DevOps Integration': 'System Engineer',
    'System Engineer - Monitoring': 'System Engineer',
    'System Engineer - Security': 'System Security Engineer',
    'System Engineer - Virtualization': 'System Engineer',
    
    # =====================================================
    # UX DESIGNER - All variations
    # =====================================================
    'Junior UX Designer': 'UX Designer',
    'Associate UX Designer': 'UX Designer',
    'Trainee UX Designer': 'UX Designer',
    'UX Intern': 'UX Designer',
    'Junior Interaction Designer': 'Interaction Designer',
    'Junior UX Analyst': 'UX Analyst',
    'UX Apprentice': 'UX Designer',
    'UX Trainee Designer': 'UX Designer',
    'UX Design Intern': 'UX Designer',
    'Senior UX Designer': 'UX Designer',
    'Lead UX Designer': 'UX Designer',
    'Principal UX Designer': 'UX Designer',
    'UX Architect': 'UX Architect',
    'UX Manager': 'UX Designer',
    'Senior Interaction Designer': 'Interaction Designer',
    'UX Strategist': 'UX Designer',
    'UX Consultant': 'UX Designer',
    'Staff UX Designer': 'UX Designer',
    'UX Design Lead': 'UX Designer',
    'Interaction Designer': 'Interaction Designer',
    'UX Analyst': 'UX Analyst',
    
    # =====================================================
    # TEST AUTOMATION / QA ENGINEER - All variations
    # =====================================================
    'Test Automation Engineer - Fresher': 'Test Automation Engineer',
    'Test Automation Engineer - Experienced': 'Test Automation Engineer',
    
    # =====================================================
    # BI ANALYST - All variations
    # =====================================================
    'BI Analyst - Fresher': 'BI Analyst',
    'BI Analyst - Experienced': 'BI Analyst',
    
    # =====================================================
    # AI ENGINEER - All variations
    # =====================================================
    'AI Engineer - Fresher': 'AI Engineer',
    'AI Engineer - Experienced': 'AI Engineer',
}


def standardize_title(title):
    """
    Standardize job titles to canonical forms.
    First clean the title, then apply mapping.
    """
    if pd.isna(title):
        return title
    
    # First clean the title (remove suffixes)
    cleaned = clean_title(title)
    
    # Then apply standardization mapping
    if cleaned in TITLE_MAPPING:
        return TITLE_MAPPING[cleaned]
    
    return cleaned


def parse_years_of_experience(years_str):
    """
    Parse YearsOfExperience strings into numeric values.
    Returns the minimum value from ranges.
    
    Examples:
    - "0-1" → 0
    - "3+" → 3
    - "5-10" → 5
    - "0–1 year" → 0  (handles en-dash)
    - "5+ years" → 5
    - "4–8 years" → 4
    - "0" → 0
    """
    if pd.isna(years_str):
        return 0
    
    years_str = str(years_str).strip()
    
    # Remove "year" or "years" suffix
    years_str = re.sub(r'\s*years?\s*$', '', years_str, flags=re.IGNORECASE)
    
    # Handle "X+" format
    match = re.match(r'^(\d+)\+', years_str)
    if match:
        return int(match.group(1))
    
    # Handle range format with hyphen or en-dash: "X-Y" or "X–Y"
    match = re.match(r'^(\d+)\s*[-–]\s*(\d+)', years_str)
    if match:
        return int(match.group(1))
    
    # Handle single number
    match = re.match(r'^(\d+)$', years_str)
    if match:
        return int(match.group(1))
    
    # Default to 0 if parsing fails
    return 0


def normalize_experience_level_by_years(years_min):
    """
    Normalize experience level based on years of experience.
    Entry-Level: 0-2 years
    Mid-Level: 2-4 years
    Senior-Level: 4+ years
    """
    if pd.isna(years_min):
        return "Entry-Level"
    
    years = int(years_min)
    
    if years <= 2:
        return "Entry-Level"
    elif years <=4:
        return "Mid-Level"
    else:
        return "Senior-Level"


def aggregate_skills(skills_list):
    """
    Aggregate skills from multiple rows into a unique set.
    Skills are semicolon-separated in the dataset.
    """
    all_skills = set()
    for skills_str in skills_list:
        if pd.notna(skills_str):
            skills = [s.strip() for s in str(skills_str).split(';') if s.strip()]
            all_skills.update(skills)
    
    # Sort and join
    return '; '.join(sorted(all_skills))


def clean_dataset(df):
    """Apply all cleaning operations to the dataset"""
    
    print("\n" + "="*60)
    print("CLEANING DATASET")
    print("="*60)
    
    # 1. Remove unused columns
    df = remove_unused_columns(df)
    
    # 2. Clean and standardize titles
    print("\nCleaning and standardizing job titles...")
    df['Title'] = df['Title'].apply(standardize_title)
    
    # Remove rows with empty titles
    before_count = len(df)
    df = df.dropna(subset=['Title'])
    df = df[df['Title'].str.strip() != '']
    after_count = len(df)
    print(f"Removed {before_count - after_count} rows with empty titles")
    
    # 3. Parse YearsOfExperience
    print("\nParsing YearsOfExperience to numeric values...")
    df['YearsOfExperience_Min'] = df['YearsOfExperience'].apply(parse_years_of_experience)
    
    # 4. Normalize ExperienceLevel based on years (Entry: 0-2, Mid: 2-4, Senior: 4+)
    print("Normalizing ExperienceLevel based on years of experience...")
    df['ExperienceLevel_Normalized'] = df['YearsOfExperience_Min'].apply(normalize_experience_level_by_years)
    
    print(f"\nCleaned dataset shape: {df.shape}")
    print(f"Unique titles: {df['Title'].nunique()}")
    
    return df


def create_aggregated_dataset(df):
    """
    Create an aggregated dataset where each unique Title + ExperienceLevel combination
    has all skills combined. This is useful for ML models that need to match skills
    to roles at different experience levels.
    """
    print("\n" + "="*60)
    print("CREATING AGGREGATED DATASET")
    print("="*60)
    
    # Group by Title and normalized ExperienceLevel
    aggregated = df.groupby(['Title', 'ExperienceLevel_Normalized']).agg({
        'Skills': aggregate_skills,
        'YearsOfExperience_Min': 'min',  # Take minimum years for the level
        'ExperienceLevel': 'first',  # Keep original experience level
        'YearsOfExperience': 'first',  # Keep original years string
    }).reset_index()
    
    # Rename columns for clarity
    aggregated = aggregated.rename(columns={
        'ExperienceLevel_Normalized': 'ExperienceLevel_Category',
        'Skills': 'All_Skills'
    })
    
    # Reorder columns
    aggregated = aggregated[['Title', 'ExperienceLevel', 'ExperienceLevel_Category', 
                             'YearsOfExperience', 'YearsOfExperience_Min', 'All_Skills']]
    
    print(f"Aggregated dataset shape: {aggregated.shape}")
    print(f"Unique title-experience combinations: {len(aggregated)}")
    
    return aggregated


def print_summary_statistics(df, aggregated):
    """Print summary statistics about the cleaned data"""
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print("\n--- Cleaned Dataset ---")
    print(f"Total records: {len(df)}")
    print(f"Unique job titles: {df['Title'].nunique()}")
    
    print("\nTop 20 Job Titles by count:")
    title_counts = df['Title'].value_counts().head(20)
    for title, count in title_counts.items():
        print(f"  {title}: {count}")
    
    print("\nExperience Level Distribution:")
    exp_counts = df['ExperienceLevel_Normalized'].value_counts()
    for level, count in exp_counts.items():
        print(f"  {level}: {count}")
    
    print("\nYears of Experience Range:")
    print(f"  Min: {df['YearsOfExperience_Min'].min()}")
    print(f"  Max: {df['YearsOfExperience_Min'].max()}")
    print(f"  Mean: {df['YearsOfExperience_Min'].mean():.2f}")
    
    print("\n--- Aggregated Dataset ---")
    print(f"Total unique role-level combinations: {len(aggregated)}")
    print(f"Unique roles: {aggregated['Title'].nunique()}")
    
    print("\nAll Unique Standardized Job Titles:")
    for i, title in enumerate(sorted(aggregated['Title'].unique()), 1):
        print(f"  {i}. {title}")


def main():
    """Main function to run the cleaning pipeline"""
    
    print("="*60)
    print("JOB DATASET CLEANING PIPELINE - Version 2")
    print("="*60)
    
    # Check if input file exists
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file not found at {INPUT_PATH}")
        print("Make sure to run this script from the project root directory.")
        return
    
    # Load data
    df = load_data()
    
    # Clean dataset
    df_cleaned = clean_dataset(df)
    
    # Create aggregated dataset
    df_aggregated = create_aggregated_dataset(df_cleaned)
    
    # Print summary statistics
    print_summary_statistics(df_cleaned, df_aggregated)
    
    # Save cleaned datasets
    print("\n" + "="*60)
    print("SAVING CLEANED DATASETS")
    print("="*60)
    
    # Save cleaned dataset (with all rows, just cleaned)
    df_cleaned.to_csv(OUTPUT_CLEANED_PATH, index=False)
    print(f"\nCleaned dataset saved to: {OUTPUT_CLEANED_PATH}")
    
    # Save aggregated dataset
    df_aggregated.to_csv(OUTPUT_AGGREGATED_PATH, index=False)
    print(f"Aggregated dataset saved to: {OUTPUT_AGGREGATED_PATH}")
    
    print("\n" + "="*60)
    print("CLEANING COMPLETE!")
    print("="*60)
    
    print("\nFiles created:")
    print(f"  1. {OUTPUT_CLEANED_PATH}")
    print("     - Original structure with cleaned titles")
    print("     - Added YearsOfExperience_Min (numeric)")
    print("     - Added ExperienceLevel_Normalized")
    print(f"\n  2. {OUTPUT_AGGREGATED_PATH}")
    print("     - Unique Title + ExperienceLevel combinations")
    print("     - Skills aggregated (all skills for each role-level)")
    print("     - Better for ML models matching skills to roles")
    
    return df_cleaned, df_aggregated


if __name__ == "__main__":
    main()
