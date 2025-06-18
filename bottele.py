import logging
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Constants ===
hasil_path = 'hasil_prediksi.csv'
NISN, IPA, IPS, PKN, MTK = range(5)
user_data_dict = {}

# === Load Model dan Scaler ===
try:
    with open('scalar_jurusan.pkl', 'rb') as f:
        loaded_data = pickle.load(f)

    if isinstance(loaded_data, tuple):
        df = loaded_data[0]
        scaler = loaded_data[1]
    else:
        df = loaded_data
        scaler = StandardScaler()
        X = df[['IPA', 'IPS', 'PKN', 'MTK']]
        scaler.fit(X)

    df['jurusan'] = ((df['IPA'] + df['MTK']) > (df['IPS'] + df['PKN'])).astype(int)
    X = df[['IPA', 'IPS', 'PKN', 'MTK']]
    y = df['jurusan']

    model = Sequential([
        Dense(8, input_dim=4, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y, epochs=50, verbose=0)

except Exception as e:
    logger.error(f"Model error: {e}")
    exit(1)

# === Start Command ===
def start(update: Update, context: CallbackContext):
    update.message.reply_text("Halo! Kirim NISN siswa:")
    return NISN

def get_nisn(update: Update, context: CallbackContext):
    nisn = update.message.text.strip()
    if os.path.exists(hasil_path):
        df = pd.read_csv(hasil_path, dtype={'NISN': str})
        if nisn in df['NISN'].values:
            update.message.reply_text("❌ NISN ini sudah pernah dipakai untuk prediksi.")
            return ConversationHandler.END
    context.user_data['nisn'] = nisn
    update.message.reply_text("Masukkan nilai IPA:")
    return IPA

def get_ipa(update: Update, context: CallbackContext):
    context.user_data['IPA'] = float(update.message.text.strip())
    update.message.reply_text("Masukkan nilai IPS:")
    return IPS

def get_ips(update: Update, context: CallbackContext):
    context.user_data['IPS'] = float(update.message.text.strip())
    update.message.reply_text("Masukkan nilai PKN:")
    return PKN

def get_pkn(update: Update, context: CallbackContext):
    context.user_data['PKN'] = float(update.message.text.strip())
    update.message.reply_text("Masukkan nilai Matematika:")
    return MTK

def get_mtk(update: Update, context: CallbackContext):
    context.user_data['MTK'] = float(update.message.text.strip())

    data = context.user_data
    input_data = pd.DataFrame([[data['IPA'], data['IPS'], data['PKN'], data['MTK']]],
                              columns=['IPA', 'IPS', 'PKN', 'MTK'])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0][0]
    jurusan = 'IPA' if pred >= 0.5 else 'IPS'

    new_entry = pd.DataFrame([{
        'NISN': data['nisn'],
        'IPA': data['IPA'],
        'IPS': data['IPS'],
        'PKN': data['PKN'],
        'MTK': data['MTK'],
        'Prediksi': jurusan
    }])

    if os.path.exists(hasil_path):
        hasil_df = pd.read_csv(hasil_path)
        hasil_df = pd.concat([hasil_df, new_entry], ignore_index=True)
    else:
        hasil_df = new_entry

    hasil_df.to_csv(hasil_path, index=False)

    update.message.reply_text(f"✅ Prediksi jurusan untuk NISN {data['nisn']}: *{jurusan}*", parse_mode="Markdown")
    return ConversationHandler.END

def cancel(update: Update, context: CallbackContext):
    update.message.reply_text("Dibatalkan.")
    return ConversationHandler.END

# === Bot Setup ===
def main():
    TOKEN = "8048564187:AAFMVQ7RORMeg1vA8p5HXN2KaB2sbblkXTU"
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            NISN: [MessageHandler(Filters.text & ~Filters.command, get_nisn)],
            IPA: [MessageHandler(Filters.text & ~Filters.command, get_ipa)],
            IPS: [MessageHandler(Filters.text & ~Filters.command, get_ips)],
            PKN: [MessageHandler(Filters.text & ~Filters.command, get_pkn)],
            MTK: [MessageHandler(Filters.text & ~Filters.command, get_mtk)],
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    dp.add_handler(conv_handler)
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
