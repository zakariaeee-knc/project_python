import mysql.connector
import os

# 🚀 CRÉER DOSSIER AVANT TOUT
os.makedirs('images', exist_ok=True)
print("✅ DOSSIER images/ créé")

# Connexion DB
db = mysql.connector.connect(host="localhost", user="root", password="2929", database="instagram_ml_real")
cursor = db.cursor()

print("🔍 DIAGNOSTIC IMAGES")
print("="*50)

# 1. MAINTENANT os.listdir marche !
print("\n📁 FICHIERS images/ trouvés:")
try:
    image_files = [f for f in os.listdir('images') if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for img in image_files:
        print(f"  ✅ {img}")
except:
    print("  📂 Aucun fichier image")

# 2. DB posts
cursor.execute("SELECT post_name, image_path FROM posts")
print("\n📊 POSTS EN BASE:")
for post_name, img_path in cursor.fetchall():
    exists = os.path.exists(img_path) if img_path else False
    print(f"  {post_name} → {img_path} {'✅' if exists else '❌'}")

# 3. AUTO-FIX
print("\n🔧 AUTO-FIX DB...")
for post_name in ['montagne_alpes', 'plage_bali']:  # Vos posts
    img_path = f"images/{post_name}.jpg"
    cursor.execute("UPDATE posts SET image_path = %s WHERE post_name = %s", (img_path, post_name))
    print(f"  🔗 {post_name} ← {img_path}")

db.commit()
cursor.close()
db.close()
print("\n✅ FIX TERMINÉ ! Lancez votre app.")
