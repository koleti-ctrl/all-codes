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
    """Translate UI labels"""
    if lang.lower() == "english":
        return text
    try:
        return GoogleTranslator(source='auto', target=lang[:2].lower()).translate(text)
    except:
        return text

def translate_to_english(text, lang):
    """Translate user input to English for matching"""
    if lang.lower() == "english" or not text.strip():
        return text
    try:
        return GoogleTranslator(source=lang[:2].lower(), target='en').translate(text)
    except:
        return text

def translate_output(text, lang):
    """Translate output to selected language"""
    if lang.lower() == "english" or not text.strip():
        return text
    try:
        return GoogleTranslator(source='auto', target=lang[:2].lower()).translate(text)
    except:
        return text

# ===============================
# Animated CSS
# ===============================
st.markdown("""
<style>
.stApp { background-color: #1e1e2f; color: white; }
.internship-card { background-color: #2e2e3f; padding:20px; margin-bottom:10px; border-radius:15px; box-shadow:0 4px 12px rgba(0,0,0,0.5); transition:transform 0.3s; }
.internship-card:hover { transform: translateY(-7px) scale(1.02); }
.internship-title { font-size:22px; font-weight:700; color:#ffcc00; margin-bottom:10px; }
.internship-detail { font-size:16px; color:#f0f0f0; margin-bottom:6px; }
.badge { display:inline-block; padding:5px 10px; margin-right:5px; border-radius:12px; font-size:12px; font-weight:bold; color:white; }
.badge-online { background-color:#27ae60; } .badge-offline { background-color:#c0392b; } .badge-skill { background-color:#2980b9; }
</style>
""", unsafe_allow_html=True)

# ===============================
# Sidebar filters
# ===============================
st.sidebar.header(translate_ui("🧑 Your Profile", "english"))

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

# ===============================
# Recommendation function
# ===============================
def recommend_internships(user_skills, sector, state, district, top_n=5):
    df_copy = df.copy()
    if sector != "Any":
        df_copy = df_copy[df_copy["Sector/Industry"].str.contains(sector, case=False, na=False)]
    if state != "Any":
        df_copy = df_copy[df_copy["State"].str.contains(state, case=False, na=False)]
    if district != "Any":
        df_copy = df_copy[df_copy["District"].str.contains(district, case=False, na=False)]
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
        df_copy = df_copy.sort_values(by="Opportunities Count", ascending=False)
    return df_copy.head(top_n)

# ===============================
# Display recommendations
# ===============================
if st.sidebar.button(translate_ui("🔍 Recommend Internships", language)):
    results = recommend_internships(skills, sector, state, district, top_n=5)
    if results.empty:
        st.warning(translate_ui("⚠️ No matching internships found. Try changing your filters.", language))
    else:
        st.subheader(translate_ui("✨ Top Recommended Internships", language))
        for _, row in results.iterrows():
            mode_class = "badge-online" if str(row["Internship Mode"]).lower() == "online" else "badge-offline"

            # Translate details to desired language
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
            </div>
            """, unsafe_allow_html=True)

            # Expander with full details
            with st.expander(translate_ui("📖 View Full Details", language)):
                st.markdown(f"""
                **Company Name:** {company_name}  
                **Sector/Industry:** {sector_name}  
                **Education (Optional):** {education if education else 'Not specified'}  
                **Internship Mode:** {row['Internship Mode']}  
                **Address:** {address}  
                **District / State:** {district_trans}, {state_trans}  
                **Opportunities:** {row['Opportunities Count']}  
                **Skills Required:** {skills_req}  
                **Role / Description:** {description}  
                """)