import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

st.write("test")

st.set_page_config(
    page_title="Nailed It — تشخيص الأظافر",
    page_icon="💅",
    layout="wide"
)

MODEL_PATH = "model/nail_model.keras"
CLASS_NAMES = ['Acral_Lentiginous_Melanoma', 'clubbing', 'healthy', 'psoriasis']
CLASS_NAMES_AR = {
    'Acral_Lentiginous_Melanoma': 'ورم ميلانيني',
    'clubbing':                   'تضخم الأظافر',
    'healthy':                    'طبيعي',
    'psoriasis':                  'صدفية الأظافر'
}
ADVICE = {
    'Acral_Lentiginous_Melanoma': ('خطر', 'يُنصح بمراجعة طبيب فوراً — هذه الحالة تستدعي تقييماً عاجلاً'),
    'clubbing':                   ('تنبيه', 'يُنصح بمراجعة طبيب — قد يكون مؤشراً لحالة داخلية'),
    'healthy':                    ('طبيعي', 'ظفرك يبدو بصحة جيدة — استمر في العناية به'),
    'psoriasis':                  ('تنبيه', 'يُنصح بمراجعة طبيب جلدية للحصول على علاج مناسب')
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

model = load_model()

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { direction: rtl; }
    [data-testid="stSidebar"] { direction: rtl; }
    .main-header {
        background: #4B1528;
        padding: 16px 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .main-header h1 { color: #F4C0D1; font-size: 22px; margin: 0; }
    .main-header p  { color: #ED93B1; font-size: 13px; margin: 0; }
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #f0e0e8;
    }
    .confidence-bar-bg {
        background: #f0f0f0;
        border-radius: 4px;
        height: 8px;
        margin: 4px 0 10px 0;
    }
    .disclaimer {
        font-size: 11px;
        color: #888;
        text-align: center;
        margin-top: 16px;
        padding: 8px;
        background: #f9f9f9;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div>
        <h1>Nailed It — تشخيص الأظافر</h1>
        <p>فحص أولي بالذكاء الاصطناعي | EfficientNetB0 | دقة 97.07%</p>
    </div>
    <div style="background:#72243E;color:#F4C0D1;font-size:11px;padding:4px 12px;border-radius:20px">نسخة تجريبية</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### القائمة")
    st.markdown("---")

    st.markdown("#### إحصائيات النموذج")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("دقة النموذج", "97.07%")
    with col2:
        st.metric("عدد الفئات", "4")

    st.markdown("---")
    st.markdown("#### نصائح التصوير")
    st.info("""
    - إضاءة طبيعية جيدة
    - ظفر واحد في الصورة
    - تجنب الظلال
    - صورة واضحة وحادة
    """)

    st.markdown("---")
    st.markdown("#### الفئات")
    for en, ar in CLASS_NAMES_AR.items():
        st.markdown(f"• **{ar}** — {en}")

col_upload, col_result = st.columns(2, gap="large")

with col_upload:
    st.markdown("### رفع الصورة")
    uploaded_file = st.file_uploader(
        "اختر صورة الظفر",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة المرفوعة", use_column_width=True)
        analyze = st.button("تحليل الصورة ←", use_container_width=True, type="primary")
    else:
        st.markdown("""
        <div style="border: 2px dashed #ED93B1; border-radius: 12px; padding: 40px;
                    text-align: center; background: #FBEAF0; color: #993556;">
            <div style="font-size: 32px; margin-bottom: 8px">💅</div>
            <div style="font-weight: 500; margin-bottom: 4px">اسحب الصورة هنا أو انقر للاختيار</div>
            <div style="font-size: 12px">JPG, PNG — حتى 10 ميجابايت</div>
        </div>
        """, unsafe_allow_html=True)
        analyze = False

with col_result:
    st.markdown("### نتيجة التحليل")

    if uploaded_file and analyze:
        with st.spinner("جاري تحليل الصورة..."):
            processed = preprocess_image(image)
            predictions = model.predict(processed)[0]

        top_idx = int(np.argmax(predictions))
        top_class = CLASS_NAMES[top_idx]
        confidence = float(predictions[top_idx]) * 100
        status, advice_text = ADVICE[top_class]

        if top_class == 'healthy':
            st.success(f"التشخيص: {CLASS_NAMES_AR[top_class]}")
        elif top_class == 'Acral_Lentiginous_Melanoma':
            st.error(f"التشخيص: {CLASS_NAMES_AR[top_class]}")
        else:
            st.warning(f"التشخيص: {CLASS_NAMES_AR[top_class]}")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("نسبة الثقة", f"{confidence:.1f}%")
        with col_m2:
            st.metric("الحالة", status)

        st.markdown("#### توزيع التصنيفات")
        for i, (cls, prob) in enumerate(zip(CLASS_NAMES, predictions)):
            pct = float(prob) * 100
            bar_color = "#72243E" if i == top_idx else "#cccccc"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;direction:rtl">
                <span style="width:160px;font-size:12px;text-align:right">{CLASS_NAMES_AR[cls]}</span>
                <div style="flex:1;background:#f0f0f0;border-radius:4px;height:8px">
                    <div style="width:{pct:.1f}%;background:{bar_color};height:8px;border-radius:4px"></div>
                </div>
                <span style="font-size:12px;width:40px">{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        if top_class == 'healthy':
            st.info(advice_text)
        elif top_class == 'Acral_Lentiginous_Melanoma':
            st.error(advice_text)
        else:
            st.warning(advice_text)

        st.markdown('<div class="disclaimer">⚠ هذا التطبيق لأغراض توعوية فقط ولا يُغني عن زيارة الطبيب المختص</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="border:1px solid #f0e0e8;border-radius:12px;padding:40px;
                    text-align:center;color:#aaa;background:#fafafa">
            <div style="font-size:28px;margin-bottom:8px">🔍</div>
            <div style="font-size:13px">ارفع صورة وانقر تحليل لرؤية النتائج</div>
        </div>
        """, unsafe_allow_html=True)