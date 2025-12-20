import os
from PIL import Image, ImageDraw

# 🚀 1. CRÉER DOSSIER images/ AUTOMATIQUE
os.makedirs('images', exist_ok=True)
print("✅ DOSSIER 'images/' CRÉÉ !")

# 2. GÉNÉRER VOTRE IMAGE
img = Image.new('RGB', (300, 300), color='#87CEEB')  # Bleu montagne
draw = ImageDraw.Draw(img)
draw.text((30, 120), "MONTAGNE", fill='white')
draw.text((50, 160), "ALPES", fill='white')

# ✅ MAINTENANT ÇA MARCHE
img.save('images/montagne_alpes.jpg', 'JPEG')
print("✅ images/montagne_alpes.jpg SAUVÉE !")
