# train_model.py
# ------------------------------------------------------------------
# شغّل الملف ده مرة واحدة بس (محلياً عندك)، هيدرب الموديل ويحفظه
# في ملف spam_model.joblib. بعدين app.py هيحمّل الملف ده مباشرة
# من غير ما يعيد التدريب كل مرة السيرفر يشتغل أو يصحى من النوم.
#
# تشغيل:
#   python train_model.py
#
# لازم يكون spam.csv موجود جنب الملف ده.
# ------------------------------------------------------------------

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "spam.csv")
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.joblib")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\n\nمقدرش ألاقي ملف spam.csv في:\n{DATA_PATH}\n"
        "حط ملف spam.csv جنب train_model.py وشغّل تاني.\n"
    )

df = pd.read_csv(DATA_PATH, encoding="latin-1")
df = df.iloc[:, :2]
df.columns = ["label", "message"]
df = df.dropna(subset=["label", "message"])

df["label"] = df["label"].str.strip().str.lower().map({"ham": 0, "spam": 1})
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

X = df["message"]
y = df["label"]

tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, test_pred) * 100, 2)

print(f"[train_model] held-out test accuracy: {accuracy}%")

# بنحفظ التوكنايزر والموديل والدقة كلهم في ملف واحد
joblib.dump({"tfidf": tfidf, "model": model, "accuracy": accuracy}, MODEL_PATH)

print(f"[train_model] الموديل اتحفظ في: {MODEL_PATH}")
print("دلوقتي ارفع spam_model.joblib ده مع app.py على GitHub (متلزمش ترفع spam.csv تاني).")
