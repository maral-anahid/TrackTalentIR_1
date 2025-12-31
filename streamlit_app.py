import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Import engines
from app.engines.text_engine import TalentRecommender
from app.engines.pose_engine import PoseAnalyzer
from app.engines.record_engine import RecordPredictor


def inject_css():
    """Inject a bit of custom CSS to make the UI look nicer.

    Streamlit does not provide a native dark theme in all versions. We embed a few CSS rules
    to tweak fonts and colours without relying on external files.
    """
    st.markdown(
        """
        <style>
        .main {
            font-family: "Vazirmatn", sans-serif;
        }
        h1, h2, h3 {
            font-weight: 700;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.4rem 1rem;
            border-radius: 5px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
            color: white;
        }
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>div>button {
            background-color: #f6f6f6;
            border-radius: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def consent_page():
    """Display the privacy consent page."""
    st.title("رضایت‌نامه و حریم خصوصی")
    st.write(
        "این اپ برای استعداد‌یابی دوومیدانی است. پیش‌فرض برنامه این است که ویدئو ذخیره نمی‌شود و فقط ویژگی‌های استخراج‌شده تحلیل می‌شوند. "
        "در صورت تمایل می‌توانید به تیم اجازه دهید داده‌ها را برای بهبود مدل‌ها ذخیره کند. برای ادامه، لطفاً تیک زیر را بزنید."
    )
    agree = st.checkbox("می‌پذیرم و ادامه می‌دهم", value=False)
    st.session_state["consent_ok"] = bool(agree)
    if agree:
        st.success("ممنون! اکنون می‌توانید از سایدبار صفحه مورد نظر را انتخاب کنید.")
    else:
        st.info("برای رفتن به صفحات دیگر، ابتدا رضایت‌نامه را تأیید کنید.")


def talent_form_page(recommender: TalentRecommender):
    """Render the talent form page for phase 1."""
    st.header("فاز ۱: فرم استعداد‌یابی")
    st.write("لطفاً اطلاعات زیر را با دقت وارد کنید تا سیستم بتواند بهترین گروه مادهٔ دوومیدانی را برای شما پیشنهاد دهد.")
    # Layout with two columns
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("سن (سال)", min_value=10, max_value=80, value=20)
        sex = st.selectbox("جنسیت", options=["مرد", "زن"])
        height = st.number_input("قد (سانتی‌متر)", min_value=100, max_value=250, value=170)
        weight = st.number_input("وزن (کیلوگرم)", min_value=30, max_value=200, value=70)
        sprint_30 = st.number_input("زمان دوی ۳۰ متر (ثانیه)", min_value=3.0, max_value=20.0, value=5.0, step=0.1)
    with col2:
        run_300 = st.number_input("زمان دوی ۳۰۰ متر (ثانیه)", min_value=30.0, max_value=300.0, value=80.0, step=1.0)
        vertical_jump = st.number_input("پرش عمودی (سانتی‌متر)", min_value=10.0, max_value=150.0, value=40.0, step=1.0)
        standing_long_jump = st.number_input("پرش طول ایستاده (سانتی‌متر)", min_value=50.0, max_value=350.0, value=160.0, step=1.0)
        plank = st.number_input("زمان پلانک (ثانیه)", min_value=10.0, max_value=600.0, value=60.0, step=5.0)
        goal_text = st.text_area("هدف یا علاقه‌مندی خود را شرح دهید", value="افزایش سرعت و استقامت")

    if st.button("پیشنهاد رشته"):
        # Build input dictionary
        input_dict = {
            "age": age,
            "sex": sex,
            "height_cm": height,
            "weight_kg": weight,
            "sprint_30m_sec": sprint_30,
            "run_300m_sec": run_300,
            "vertical_jump_cm": vertical_jump,
            "standing_long_jump_cm": standing_long_jump,
            "plank_sec": plank,
            "goal_text": goal_text,
        }
        with st.spinner("در حال پردازش..."):
            results = recommender.predict(input_dict)

        # Display results in a bar chart
        st.subheader("نتایج پیشنهادی")
        df_results = pd.DataFrame(results, columns=["گروه ماده", "احتمال"])
        chart = (
            alt.Chart(df_results)
            .mark_bar(color="#4F8BF9")
            .encode(
                x=alt.X("گروه ماده:N", sort="-y"),
                y=alt.Y("احتمال:Q", title="احتمال"),
                tooltip=["گروه ماده", "احتمال"],
            )
            .properties(width=500, height=300)
        )
        st.altair_chart(chart, use_container_width=True)
        st.write("بر اساس اطلاعاتی که وارد کرده‌اید، سه گروه مادهٔ مناسب برای شما در جدول بالا نمایش داده شده‌است.")


def pose_analysis_page(analyzer: PoseAnalyzer):
    """Render the pose analysis page for phase 2."""
    st.header("فاز ۲: تحلیل ویدئو")
    st.write(
        "در این بخش می‌توانید ویدئوی کوتاهی از حرکات استاندارد (اسکوات، پرش یا دو کوتاه) خود را بارگذاری کنید تا سیستم کیفیت حرکت و آمادگی حرکتی شما را ارزیابی کند."
    )
    video_file = st.file_uploader("ویدئو را انتخاب کنید", type=["mp4", "mov", "m4v"])
    if video_file is not None:
        st.video(video_file)
        if st.button("شروع تحلیل"):
            with st.spinner("در حال تحلیل ویدئو..."):
                # Save the uploaded file temporarily
                video_path = "/tmp/uploaded_video.mp4"
                with open(video_path, "wb") as f:
                    f.write(video_file.getbuffer())
                result = analyzer.analyze(video_path)
            # Display result
            st.subheader("نتیجهٔ تحلیل")
            score = result.get("score", 0)
            remarks = result.get("remarks", "توضیحی وجود ندارد.")
            st.metric(label="امتیاز آمادگی حرکتی", value=f"{score:.2f}", delta=None)
            st.info(remarks)
            st.success("تحلیل به پایان رسید.")
    else:
        st.info("ویدئویی انتخاب نشده است.")


def record_predictor_page(predictor: RecordPredictor):
    """Render the record prediction page for phase 3."""
    st.header("فاز ۳: پیش‌بینی رکورد آینده")
    st.write(
        "برای پیش‌بینی رکورد یا رتبهٔ آینده، رکوردهای اخیر خود را وارد کنید. "
        "این ابزار آزمایشی است و برای توسعه و تحقیق ایجاد شده‌است."
    )
    # Input: best personal record and average of last 3 competitions
    pr_time = st.number_input("بهترین رکورد شخصی (زمان/ثانیه یا مسافت/متر)", min_value=1.0, max_value=1000.0, value=12.0, step=0.1)
    last1 = st.number_input("رکورد مسابقهٔ اخیر", min_value=1.0, max_value=1000.0, value=13.0, step=0.1)
    last2 = st.number_input("رکورد مسابقهٔ دوم", min_value=1.0, max_value=1000.0, value=13.5, step=0.1)
    last3 = st.number_input("رکورد مسابقهٔ سوم", min_value=1.0, max_value=1000.0, value=14.0, step=0.1)
    if st.button("پیش‌بینی رکورد"):
        input_dict = {
            "pr_time": pr_time,
            "last1": last1,
            "last2": last2,
            "last3": last3,
        }
        with st.spinner("در حال پیش‌بینی..."):
            result = predictor.predict(input_dict)
        predicted_time = result.get("predicted_time", None)
        notes = result.get("notes", "")
        if predicted_time is not None:
            st.subheader("پیش‌بینی رکورد")
            st.success(f"رکورد پیش‌بینی‌شده: {predicted_time:.2f}")
            st.info(notes)
        else:
            st.error("مدل پیش‌بینی در دسترس نیست.")


def main():
    # Configure page
    st.set_page_config(page_title="TrackTalentIR", page_icon="🏃", layout="wide")
    inject_css()

    # Instantiate engines
    recommender = TalentRecommender(model_path="models/text_model.pkl")
    analyzer = PoseAnalyzer()
    predictor = RecordPredictor()

    # Sidebar navigation
    st.sidebar.title("صفحات")
    page_names = ["رضایت‌نامه", "فاز ۱: فرم استعداد‌یابی", "فاز ۲: تحلیل ویدئو", "فاز ۳: پیش‌بینی رکورد"]
    selected_page = st.sidebar.radio("انتخاب صفحه", page_names)

    # Require consent for other pages
    if "consent_ok" not in st.session_state:
        st.session_state["consent_ok"] = False

    if selected_page == "رضایت‌نامه":
        consent_page()
    else:
        if not st.session_state["consent_ok"]:
            st.error("لطفاً ابتدا رضایت‌نامه را در صفحهٔ اول تأیید کنید.")
            consent_page()
        else:
            if selected_page == "فاز ۱: فرم استعداد‌یابی":
                talent_form_page(recommender)
            elif selected_page == "فاز ۲: تحلیل ویدئو":
                pose_analysis_page(analyzer)
            elif selected_page == "فاز ۳: پیش‌بینی رکورد":
                record_predictor_page(predictor)


if __name__ == "__main__":
    main()