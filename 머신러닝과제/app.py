# 1. 필요한 라이브러리
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import uniform
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# 2. 데이터 불러오기
train_data = pd.read_csv("ratings_train.txt", sep='\t')
test_data = pd.read_csv("ratings_test.txt", sep='\t')

# 3. 데이터 전처리 함수 정의
def preprocess(text):
    text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣ\s]", "", str(text))
    return text.strip()

train_data['document'] = train_data['document'].fillna("").apply(preprocess)
test_data['document'] = test_data['document'].fillna("").apply(preprocess)

# 4. 벡터화 (TF-IDF)
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_data['document'])
y_train = train_data['label']
X_test = vectorizer.transform(test_data['document'])
y_test = test_data['label']

# 5. RandomizedSearchCV로 하이퍼파라미터 튜닝
param_dist = {
    'C': uniform(0.01, 10),
    'solver': ['liblinear', 'saga']
}

logreg = LogisticRegression(max_iter=1000)
rand_search = RandomizedSearchCV(logreg, param_distributions=param_dist, n_iter=20, 
                                 scoring='accuracy', cv=5, random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)

model = rand_search.best_estimator_
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 6. Streamlit 앱
def main():
    st.title("🎬 영화 리뷰 감성 분석")
    menu = st.sidebar.selectbox("메뉴를 선택하세요", ["리뷰 분석", "EDA", "모델 정보"])

    if menu == "리뷰 분석":
        user_input = st.text_input("리뷰를 입력하세요:")
        if user_input:
            processed = preprocess(user_input)
            vectorized = vectorizer.transform([processed])
            prediction = model.predict(vectorized)[0]
            prob = model.predict_proba(vectorized)[0][prediction]
            if prediction == 1:
                st.success(f"긍정 리뷰입니다! 😊 (확률: {prob:.2f})")
            else:
                st.error(f"부정 리뷰입니다. 😞 (확률: {prob:.2f})")

    elif menu == "EDA":
        st.subheader("데이터 EDA")

        # 한글 폰트 설정
        import matplotlib.font_manager as fm
        try:
            plt.rcParams['font.family'] = 'Malgun Gothic'
        except:
            try:
                plt.rcParams['font.family'] = 'AppleGothic'
            except:
                st.warning("한글 폰트 설정에 실패했습니다.")
        plt.rcParams['axes.unicode_minus'] = False

        st.write("✅ 데이터 개수:", train_data.shape)
        st.write("✅ 결측치 개수:", train_data.isnull().sum())

        all_words = ' '.join(train_data['document']).split()
        common_words = Counter(all_words).most_common(20)
        st.write("✅ 가장 많이 등장한 단어 (상위 20개):")
        st.dataframe(pd.DataFrame(common_words, columns=["단어", "빈도수"]))

        train_data['length'] = train_data['document'].apply(len)
        fig2, ax2 = plt.subplots()
        ax2.hist(train_data['length'], bins=50, color='skyblue')
        ax2.set_title("리뷰 길이 분포")
        ax2.set_xlabel("글자 수")
        ax2.set_ylabel("리뷰 개수")
        st.pyplot(fig2)

    elif menu == "모델 정보":
        st.subheader("모델 성능 및 하이퍼파라미터")
        st.write("✅ 테스트 정확도:", accuracy)
        st.write("✅ 최적 하이퍼파라미터:", rand_search.best_params_)
        st.text("📋 분류 리포트:")
        st.text(classification_report(y_test, y_pred, target_names=["부정", "긍정"]))

if __name__ == "__main__":
    main()
