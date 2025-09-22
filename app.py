import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
import time

# ===============================
# Page setup
# ===============================
st.set_page_config(page_title="🌐 SkillSync Internship Recommender", layout="wide")
st.title("🤖 SkillSync – AI-powered Internship Recommender")

# ===============================
# Background & Sidebar Style
# ===============================
st.markdown("""
<style>
.stApp {
    background-image: url('https://images.unsplash.com/photo-1504384308090-c894fdcc538d');
    background-size: cover;
    background-attachment: fixed;
    color: white;
}
section[data-testid="stSidebar"] {
    background-image: url('https://images.unsplash.com/photo-1518770660439-4636190af475');
    background-size: cover;
    background-attachment: fixed;
    color: white;
}
.internship-card {
    background-color: rgba(0, 0, 0, 0.7);
    padding:20px;
    margin-bottom:15px;
    border-radius:15px;
    box-shadow:0 4px 12px rgba(0,0,0,0.6);
    transition:transform 0.3s;
}
.internship-card:hover {
    transform: translateY(-7px) scale(1.02);
}
.internship-title {
    font-size:22px;
    font-weight:700;
    color:#ffcc00;
    margin-bottom:10px;
}
.internship-detail {
    font-size:16px;
    color:#f0f0f0;
    margin-bottom:6px;
}
.badge {
    display:inline-block;
    padding:5px 10px;
    margin-right:5px;
    border-radius:12px;
    font-size:12px;
    font-weight:bold;
    color:white;
}
.badge-online { background-color:#27ae60; }
.badge-offline { background-color:#c0392b; }
.badge-skill { background-color:#2980b9; }
</style>
""", unsafe_allow_html=True)

# ===============================
# Load default CSV
# ===============================
CSV_FILE = "internships.csv"
try:
    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()  # Strip any spaces in column names
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
    translate_ui("📝 Preferred Mode", language),
    options=["Any", "Online", "Offline"]
)
mode = translate_to_english(mode_input, language)

# ===============================
# Recommendation function
# ===============================
def recommend_internships(user_skills, sector, state, district, mode, top_n=5):
    if not user_skills.strip():
        return pd.DataFrame()  # Must provide skills

    df_copy = df.copy()

    if sector != "Any":
        df_copy = df_copy[df_copy["Sector/Industry"].str.contains(sector, case=False, na=False)]
    if state != "Any":
        df_copy = df_copy[df_copy["State"].str.contains(state, case=False, na=False)]
    if district != "Any":
        df_copy = df_copy[df_copy["District"].str.contains(district, case=False, na=False)]
    if mode != "Any":
        df_copy = df_copy[df_copy["Internship"].str.contains(mode, case=False, na=False)]  # Assuming 'Internship' column describes mode

    if df_copy.empty:
        return pd.DataFrame()

    # Skills similarity
    vectorizer = TfidfVectorizer()
    skill_matrix = vectorizer.fit_transform(df_copy["Required Skills"].fillna("").astype(str))
    user_vector = vectorizer.transform([user_skills])
    similarity_scores = cosine_similarity(user_vector, skill_matrix).flatten()
    df_copy["Match Score"] = similarity_scores

    df_copy = df_copy.sort_values(by="Match Score", ascending=False)
    return df_copy.head(top_n)

# ===============================
# Display recommendations
# ===============================
if st.sidebar.button(translate_ui("🔍 Recommend Internships", language), key="recommend_button"):
    if not skills.strip():
        st.warning(translate_ui("⚠️ Please enter your skills to get recommendations!", language))
    else:
        with st.spinner("⚡ Finding the best internships for you... Please wait! 🚀"):
            time.sleep(2)
            results = recommend_internships(skills, sector, state, district, mode, top_n=5)

        if results.empty:
            st.warning(translate_ui("⚠️ No matching internships found. Try changing your filters.", language))
        else:
            st.subheader(translate_ui("✨ Top Recommended Internships", language))
            for idx, row in results.iterrows():
                company_name = row["Company"]
                internship_name = row["Internship"]
                sector_name = translate_output(row["Sector/Industry"], language)
                skills_req = translate_output(row["Required Skills"], language)
                address = translate_output(row["Address"], language)
                description = translate_output(row.get("Description", "No description available"), language)
                district_trans = translate_output(row["District"], language)
                state_trans = translate_output(row["State"], language)
                last_date = translate_output(str(row.get("Last Date to Register", "Not specified")), language)
                duration = translate_output(str(row.get("Duration", "Not specified")), language)

                mode_class = "badge-online" if "online" in internship_name.lower() else "badge-offline"

                # Main card
                st.markdown(f"""
                <div class="internship-card">
                    <div class="internship-title">{company_name} - {internship_name} ({sector_name})</div>
                    <div class="internship-detail">📍 {district_trans}, {state_trans}</div>
                    <div class="internship-detail">📝 Mode: <span class="badge {mode_class}">{internship_name}</span></div>
                    <div class="internship-detail">💼 Skills: <span class="badge badge-skill">{skills_req}</span></div>
                    <div class="internship-detail">🕒 Duration: {duration}</div>
                    <div class="internship-detail">📅 Last Date: {last_date}</div>
                    <div class="internship-detail">🏢 Address: {address}</div>
                    <div class="internship-detail">📝 Role / Description: {description}</div>
                    <br>
                    <form action="{row.get('Application Link','#')}" method="get" target="_blank">
                        <button style="
                            background-color:#3182ce; 
                            color:white; 
                            border:none; 
                            padding:8px 16px; 
                            border-radius:8px; 
                            cursor:pointer;
                            font-weight:bold;
                        ">✅ Apply</button>
                    </form>
                </div>
                """, unsafe_allow_html=True)
