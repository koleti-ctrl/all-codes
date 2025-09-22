import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

# ===============================
# Page setup
# ===============================
st.set_page_config(page_title="🌐 SkillSync Internship Recommender", layout="wide")
st.title("🌐 AI-powered Internship Recommender - SkillSync")

# ===============================
# Load default CSV
# ===============================
CSV_FILE = "internships.csv"
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    st.error("⚠️ Default CSV not found! Put 'internships.csv' in the app folder.")
    st.stop()

# ===============================
# Translation helpers
# ===============================
def translate_ui(text, lang):
    if lang.lower() == "english":
        return text
    try:
        return GoogleTranslator(source='auto', target=lang[:2].lower()).translate(text)
    except:
        return text

def translate_to_english(text, lang):
    if lang.lower() == "english" or not text.strip():
        return text
    try:
        return GoogleTranslator(source=lang[:2].lower(), target='en').translate(text)
    except:
        return text

def translate_output(text, lang):
    if lang.lower() == "english" or not text.strip():
        return text
    try:
        return GoogleTranslator(source='auto', target=lang[:2].lower()).translate(text)
    except:
        return text

# ===============================
# CSS for attractive cards
# ===============================
st.markdown("""
<style>
/* App background with AI image + gradient overlay */
.stApp {
    background-image: url('https://images.unsplash.com/photo-1612831662681-cf3f5cb2ef57?auto=format&fit=crop&w=1600&q=80');
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
    color: #f0f0f0;
}

/* Sidebar with semi-transparent overlay */
.css-1d391kg {
    background-color: rgba(30, 31, 47, 0.85);
    color: #f0f0f0;
}

/* Internship cards */
.internship-card {
    background-color: rgba(255, 255, 255, 0.08);
    padding: 20px;
    margin-bottom: 15px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    transition: transform 0.3s;
}
.internship-card:hover {
    transform: translateY(-7px) scale(1.02);
}

/* Titles and badges */
.internship-title { font-size:22px; font-weight:700; color:#00ffd5; }
.internship-detail { font-size:16px; color:#f0f0f0; margin-bottom:6px; }
.badge-online { background-color:#27ae60; }
.badge-offline { background-color:#c0392b; }
.badge-skill { background-color:#2980b9; }
.badge-demand { background-color:#e67e22; }
.badge-top { background-color:#16a085; }
.badge-trend { background-color:#d35400; }
</style>
""", unsafe_allow_html=True)

# ===============================
# Sidebar filters
# ===============================
st.sidebar.header("🧑 Your Profile")

language = st.sidebar.radio("🌐 Choose Language:", ["English", "Hindi", "Telugu"])

education_input = st.sidebar.text_input(translate_ui("🎓 Your Education (optional)", language))
education = translate_to_english(education_input, language)

skills_input = st.sidebar.text_input(translate_ui("💼 Your Skills (comma-separated)", language))
skills = translate_to_english(skills_input, language)

sector_input = st.sidebar.selectbox(
    translate_ui("🏢 Preferred Sector", language),
    options=["Any"] + sorted(df["Sector/Industry"].dropna().unique())
)
sector = translate_to_english(sector_input, language)

state_input = st.sidebar.selectbox(
    translate_ui("🌍 Preferred State", language),
    options=["Any"] + sorted(df["State"].dropna().unique())
)
state = translate_to_english(state_input, language)

if state != "Any":
    districts = df[df["State"] == state]["District"].dropna().unique().tolist()
    district_input = st.sidebar.selectbox(
        translate_ui("📍 Preferred District", language),
        options=["Any"] + sorted(districts)
    )
    district = translate_to_english(district_input, language)
else:
    district = "Any"

mode_input = st.sidebar.selectbox(
    translate_ui("💻 Preferred Mode", language),
    ["Any", "Online", "Offline"]  # Removed Remote
)

if st.sidebar.button(translate_ui("🔍 Recommend Internships", language)):
    # Place spinner here
    with st.spinner("🤖 Crunching data for your dream internships... just a sec!"):
        results = recommend_internships(skills, sector, state, district, mode, top_n=5)

    # The rest of your code for displaying results
    if results.empty:
        st.warning(translate_ui("⚠️ No matching internships found. Try changing your filters.", language))
    else:
        ...

# ===============================
# Recommendation function
# ===============================
def recommend_internships(user_skills, sector, state, district, mode, top_n=5):
    df_copy = df.copy()
    if sector != "Any":
        df_copy = df_copy[df_copy["Sector/Industry"].str.contains(sector, case=False, na=False)]
    if state != "Any":
        df_copy = df_copy[df_copy["State"].str.contains(state, case=False, na=False)]
    if district != "Any":
        df_copy = df_copy[df_copy["District"].str.contains(district, case=False, na=False)]
    if mode != "Any":
        df_copy = df_copy[df_copy["Internship Mode"].str.contains(mode, case=False, na=False)]
    if df_copy.empty:
        return pd.DataFrame()

    if user_skills.strip():
        vectorizer = TfidfVectorizer()
        skill_matrix = vectorizer.fit_transform(df_copy["Required Skills"].fillna("").astype(str))
        user_vector = vectorizer.transform([user_skills])
        similarity_scores = cosine_similarity(user_vector, skill_matrix).flatten()
        df_copy["Match Score"] = similarity_scores
        df_copy = df_copy.sort_values(by=["Match Score", "Opportunities Count"], ascending=False)
    else:
        df_copy["Match Score"] = 0.0
        df_copy = df_copy.sort_values(by="Opportunities Count", ascending=False)

    return df_copy.head(top_n)

# ===============================
# Display recommendations
# ===============================
if st.sidebar.button(translate_ui("🔍 Recommend Internships", language)):
    results = recommend_internships(skills, sector, state, district, mode_input, top_n=5)
    if results.empty:
        st.warning(translate_ui("⚠️ No matching internships found. Try changing your filters.", language))
    else:
        st.subheader(translate_ui("✨ Top Recommended Internships", language))
        company_counts = df["Company Name"].value_counts()

        for _, row in results.iterrows():
            mode_class = "badge-online" if str(row["Internship Mode"]).lower() == "online" else (
                "badge-remote" if str(row["Internship Mode"]).lower() == "remote" else "badge-offline"
            )

            # Badges
            badges = []
            if row["Opportunities Count"] > 50:
                badges.append('<span class="badge badge-demand">🔥 High in Demand</span>')
            if company_counts[row["Company Name"]] > 3:
                badges.append('<span class="badge badge-top">🏆 Top Recruiter</span>')
            if any(skill.lower() in str(row["Required Skills"]).lower() for skill in ["python", "ml", "ai"]):
                badges.append('<span class="badge badge-trend">⭐ Trending Skill</span>')

            badge_str = " ".join(badges)

            # Translate details
            company_name = row["Company Name"]
            sector_name = translate_output(row["Sector/Industry"], language)
            skills_req = translate_output(row["Required Skills"], language)
            address = translate_output(row["Address"], language)
            description = translate_output(row.get("Description", "No description available"), language)
            district_trans = translate_output(row["District"], language)
            state_trans = translate_output(row["State"], language)

            # Main card
            st.markdown(f"""
            <div class="internship-card">
                <div class="internship-title">{company_name} - {sector_name}</div>
                <div class="internship-detail">📍 {district_trans}, {state_trans}</div>
                <div class="internship-detail">📝 Mode: <span class="badge {mode_class}">{row['Internship Mode']}</span></div>
                <div class="internship-detail">💼 Skills: <span class="badge badge-skill">{skills_req}</span></div>
                <div>{badge_str}</div>
            </div>
            """, unsafe_allow_html=True)

            # Expander with Apply button
            with st.expander(translate_ui("📖 View Full Details", language)):
                st.markdown(f"""
                **Company Name:** {company_name}  
                **Sector/Industry:** {sector_name}  
                **Education:** {education if education else 'Not specified'}  
                **Internship Mode:** {row['Internship Mode']}  
                **Address:** {address}  
                **District / State:** {district_trans}, {state_trans}  
                **Opportunities:** {row['Opportunities Count']}  
                **Skills Required:** {skills_req}  
                **Role / Description:** {description}  
                """)
                if st.button(f"✅ Apply to {company_name}", key=company_name+str(row["Opportunities Count"])):
                    st.success("📩 Application submitted successfully!")






