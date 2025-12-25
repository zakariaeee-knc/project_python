from PIL import Image, ImageDraw
import os

# 🚀 CRÉER DOSSIER AUTOMATIQUE
os.makedirs('images', exist_ok=True)  # ✅ CRÉE images/ si absent

print("📁 Dossier 'images/' créé ✅")

# Créer image
img = Image.new('RGB', (400, 400), color='#87CEEB')  # Ciel bleu
draw = ImageDraw.Draw(img)
draw.text((50, 180), "MONTAGNE", fill='white')
draw.text((80, 220), "ALPES", fill='white')

# ✅ SAUVEGARDE MAINTENANT FONCTIONNE
img.save('images/code_python.jpg', 'JPEG', quality=95)
print("✅ images/montagne_alpes.jpg sauvée !")