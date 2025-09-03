import os
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

# Ensure 'images' directory exists
os.makedirs("images", exist_ok=True)

# Load the cleaned data
df = pd.read_parquet("data_processed/data_jobs_cleaned.parquet")

# =======Monthly Job Postings Trend ==============

# Filter for data jobs postings in the United States
df_US = df[df['job_country'] == 'United States'].copy()
# Extract month and create a new column for better readability
df_US['month'] = df_US['job_posted_date'].dt.strftime('%b')  # e.g., Jan, Feb
df_US['month_num'] = df_US['job_posted_date'].dt.month       # numeric for sorting

# Group by month and count postings
monthly_trends = (
    df_US.groupby(['month_num','month'])
    .size()
    .reset_index(name='postings')
    .sort_values('month_num')
)

# Plot monthly trend
plt.figure(figsize=(10,6))
sns.lineplot(data=monthly_trends, x='month', y='postings', marker='o', linewidth=2.5, color='teal')
plt.title("Monthly Data Job Postings in the U.S. (2023)", fontsize=14, weight='bold')
plt.xlabel("Month")
plt.ylabel("Number of Postings")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.savefig("images/monthly_job_postings_trend.png", dpi = 150, bbox_inches='tight')
plt.close()

# ==========Top 10 Job Titles in Data Job Postings ============

# Prepare the data
top_roles = df['job_title_short'].value_counts().head(10).sort_values(ascending=False)

# Set seaborn style
sns.set_theme(style='whitegrid')

# Set up figure
fig, ax = plt.subplots(figsize=(10, 6))

# Create horizontal bar plot
bars = sns.barplot(x=top_roles.values, y=top_roles.index, palette='crest', ax=ax, hue = top_roles.index, legend = False)


# Customization
ax.set_title('Top 10 Job Titles in Data Job Postings', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Postings')
ax.set_ylabel('')
sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.savefig("images/top_10_job_titles.png", dpi=150, bbox_inches='tight')
plt.close()

# ========== Top 5 skills for the Top 3 Job Roles

# Define top roles and their teal shades
top_roles = ["Data Analyst", "Data Engineer", "Data Scientist"]
role_colors = {
    "Data Analyst": "#20B2AA",  # Light Teal
    "Data Engineer": "#008080", # Medium Teal
    "Data Scientist": "#004C4C" # Dark Teal
}

# Filter and explode
df_top_roles = df_US[df_US['job_title_short'].isin(top_roles)].copy()
df_top_roles = df_top_roles.explode('job_skills')

# Count skills per role
skill_counts = (
    df_top_roles.groupby(['job_title_short','job_skills'])
    .size()
    .reset_index(name='count')
)

# Keep top 5 per role
top5_per_role = (
    skill_counts.sort_values(['job_title_short','count'], ascending=[True, False])
    .groupby('job_title_short', group_keys=False)
    .head(5)
)

# ----- Small multiples: one subplot per role -----
sns.set_style("ticks")
fig, axes = plt.subplots(nrows=len(top_roles), ncols=1, figsize=(12, 9), sharex=True)

if len(top_roles) == 1:
    axes = [axes]  # ensure iterable if only one role

max_count = top5_per_role['count'].max()

for ax, role in zip(axes, top_roles):
    sub = (top5_per_role[top5_per_role['job_title_short'] == role]
           .sort_values('count', ascending=True))  # ascending for horizontal bars
    sns.barplot(
        data=sub,
        x='count',
        y='job_skills',
        ax=ax,
        color=role_colors[role]  # <- each role gets its own teal shade
    )
    ax.set_title(role, loc="left", fontsize=12, weight="bold")
    ax.set_ylabel("")  # cleaner
    ax.set_xlabel("Number of Postings")

    # Value labels
    for p in ax.patches:
        w = p.get_width()
        ax.text(w + max_count * 0.01, p.get_y() + p.get_height()/2,
                f"{int(w):,}", va="center", fontsize=9)

fig.suptitle("Top 5 Skills per Role (U.S.) — Counts", fontsize=14, weight="bold", y=0.98)
plt.tight_layout()
plt.savefig("images/top_5_skills_top3_role.png", dpi=150, bbox_inches='tight')
plt.close()

# =======Salary Distribution by Job Title =================

# Prepare aggregated data
# Count top N job titles
top_n = 10
top_jobs = df_US['job_title_short'].value_counts().head(top_n).index

# Filter and aggregate salary by job title
df_roles = df_US[df_US['job_title_short'].isin(top_jobs)]
df_grouped = df_roles.groupby('job_title_short')['salary_year_avg'].median().sort_values(ascending=True).to_frame().reset_index()

# Generate box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_roles[df_roles['job_title_short'].isin(top_jobs)], 
            x='salary_year_avg', y='job_title_short', palette='Set2', hue = 'job_title_short', legend = False)

plt.title('Salary Distribution by Job Title (Top 10)')
plt.xlabel('Yearly Salary (USD)')
plt.ylabel('')
plt.tight_layout()
plt.savefig("images/salary_distribution_by_job_title.png", dpi=150, bbox_inches='tight')
plt.close()

# ============ Top Paying skills for Data Analysts ====================

# Filtering for Data Analyst Roles in the U.S
df_DA = df_US[df_US['job_title'].str.contains('data analyst', case=False, na=False)]

# Drop nulls
df_DA = df_DA.dropna(subset=['salary_year_avg', 'job_skills'])

# Explode skills
df_DA_exploded = df_DA.explode('job_skills')
df_DA_exploded['job_skills'] = df_DA_exploded['job_skills'].str.strip().str.lower()


# Group by skill and get median salary
df_analyst_skill_salary = (
    df_DA_exploded
    .groupby('job_skills')['salary_year_avg']
    .median()
    .sort_values(ascending=False)
    .reset_index()
)

# Filter to top 10 highest-paid skills
df_top_analyst_skills = df_analyst_skill_salary.head(10)

# Plot

plt.figure(figsize=(10, 6))
sns.barplot(data=df_top_analyst_skills, 
            x='salary_year_avg', 
            y='job_skills', 
            palette='crest',
            hue='job_skills', 
            legend=False)

plt.title('Highest Paying Skills for Data Analysts (U.S.)', fontsize=14)
plt.xlabel('Median Yearly Salary (USD)')
plt.ylabel('Skill')
plt.tight_layout()
plt.savefig("images/highest_paying_skills_data_analysts.png", dpi=150, bbox_inches='tight')
plt.close()

print("Charts saved in 'images' directory.")