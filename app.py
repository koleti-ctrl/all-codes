import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
# Load CSV
# ===============================
CSV_FILE = "internships.csv"
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    st.error("⚠️ Default CSV not found! Put 'internships.csv' in the app folder.")
    st.stop()

# Ensure Opportunities is numeric
df["Opportunities"] = pd.to_numeric(df["Opportunities"], errors='coerce').fillna(0).astype(int)

# Strip column spaces
df.columns = df.columns.str.strip()

# ===============================
# Sidebar filters
# ===============================
st.sidebar.header("🧑 Your Profile")

skills = st.sidebar.text_input("💼 Your Skills (comma-separated)")
sector = st.sidebar.selectbox("🏢 Preferred Sector", options=["Any"] + sorted(df["Sector/Industry"].dropna().unique()))
state = st.sidebar.selectbox("🌍 Preferred State", options=["Any"] + sorted(df["State"].dropna().unique()))

if state != "Any":
    districts = df[df["State"] == state]["District"].dropna().unique().tolist()
    district = st.sidebar.selectbox("📍 Preferred District", options=["Any"] + sorted(districts))
else:
    district = "Any"

mode = st.sidebar.selectbox("📝 Preferred Mode", options=["Any", "Online", "Offline"])

# ===============================
# Recommendation function
# ===============================
def recommend_internships(user_skills, sector, state, district, mode, top_n=5):
    df_copy = df.copy()

    # Filters
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

    # Skill similarity using TF-IDF
    if user_skills.strip():
        vectorizer = TfidfVectorizer()
        skill_matrix = vectorizer.fit_transform(df_copy["Required Skills"].fillna("").astype(str))
        user_vector = vectorizer.transform([user_skills])
        similarity_scores = cosine_similarity(user_vector, skill_matrix).flatten()
        df_copy["Match Score"] = similarity_scores
        df_copy = df_copy.sort_values(by=["Match Score", "Opportunities"], ascending=False)
    else:
        df_copy = df_copy.sort_values(by="Opportunities", ascending=False)

    return df_copy.head(top_n)

# ===============================
# Display recommendations
# ===============================
if st.sidebar.button("🔍 Recommend Internships"):
    if not skills.strip():
        st.warning("⚠ Please enter your skills to get recommendations!")
    else:
        with st.spinner("⚡ Finding the best internships for you..."):
            time.sleep(1)
            results = recommend_internships(skills, sector, state, district, mode, top_n=5)

        if results.empty:
            st.warning("⚠ No matching internships found. Try changing your filters.")
        else:
            st.subheader("✨ Top Recommended Internships")
            for idx, row in results.iterrows():
                company_name = row["Company"]
                internship_title = row["Internship"]
                sector_name = row["Sector/Industry"]
                skills_req = row["Required Skills"]
                address = row["Address"]
                opportunities = int(row["Opportunities"])
                duration = row["Duration"]
                district_trans = row["District"]
                state_trans = row["State"]

                # Badges
                mode_class = "badge-online" if str(row.get("Internship Mode", "Offline")).lower() == "online" else "badge-offline"
                high_recruiting = '<span style="background-color:#e67e22;color:white;padding:4px 8px;border-radius:8px;font-size:12px;font-weight:bold;">🔥 High Recruiting</span>' if opportunities >= 10 else ""

                # Main card
                st.markdown(f"""
                <div class="internship-card">
                    <div class="internship-title">{company_name} - {internship_title} {high_recruiting}</div>
                    <div class="internship-detail">📍 {district_trans}, {state_trans}</div>
                    <div class="internship-detail">📝 Mode: <span class="badge {mode_class}">{row.get('Internship Mode', 'Offline')}</span></div>
                    <div class="internship-detail">💼 Skills: <span class="badge badge-skill">{skills_req}</span></div>
                    <div class="internship-detail">🕒 Duration: {duration}</div>
                    <div class="internship-detail">📊 Opportunities: {opportunities}</div>
                </div>
                """, unsafe_allow_html=True)

                # Expander with apply button
                expander_key = f"expander_{idx}"
                button_key = f"apply_{idx}"
                with st.expander("📖 View Full Details", expanded=False, key=expander_key):
                    st.markdown(f"""
                    **Company:** {company_name}  
                    **Internship:** {internship_title}  
                    **Sector/Industry:** {sector_name}  
                    **Skills Required:** {skills_req}  
                    **Opportunities:** {opportunities}  
                    **Duration:** {duration}  
                    **Address:** {address}  
                    **District / State:** {district_trans}, {state_trans}  
                    """)
                    if st.button("✅ Apply", key=button_key):
                        st.success(f"You chose to apply for {company_name} 🎉")

