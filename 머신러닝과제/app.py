# 1. 필요한 라이브러리
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# 2. 데이터 불러오기
train_data = pd.read_csv("ratings_train.txt", sep='\t')
test_data = pd.read_csv("ratings_test.txt", sep='\t')

# 3. 데이터 전처리 함수 정의
def preprocess(text):
    text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣ\s]", "", str(text))  # 한글, 공백만 남기기
    return text.strip()

train_data['document'] = train_data['document'].fillna("").apply(preprocess)
test_data['document'] = test_data['document'].fillna("").apply(preprocess)

# 4. 벡터화 (TF-IDF)
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_data['document'])
y_train = train_data['label']
X_test = vectorizer.transform(test_data['document'])
y_test = test_data['label']

# 5. 하이퍼파라미터 튜닝 (XGBoost)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.3]
}
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
grid = GridSearchCV(xgb, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
model = grid.best_estimator_
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
            plt.rcParams['axes.unicode_minus'] = False
        except:
            try:
                plt.rcParams['font.family'] = 'AppleGothic'
                plt.rcParams['axes.unicode_minus'] = False
            except:
                try:
                    plt.rcParams['font.family'] = 'NanumGothic'
                    plt.rcParams['axes.unicode_minus'] = False
                except:
                    st.warning("한글 폰트 설정에 실패했습니다.")

        st.write("✅ 데이터 개수:")
        st.write(train_data.shape)

        st.write("✅ 결측치 개수:")
        st.write(train_data.isnull().sum())

        # 가장 많이 등장한 단어
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
        st.write("✅ 최적 하이퍼파라미터:", grid.best_params_)
        st.text("📋 분류 리포트:")
        st.text(classification_report(y_test, y_pred, target_names=["부정", "긍정"]))

if __name__ == "__main__":
    main()
