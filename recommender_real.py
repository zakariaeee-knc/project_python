import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw  # pip install Pillow
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import threading
import warnings
warnings.filterwarnings('ignore')

class InstagramRecommenderReal:
    def __init__(self):
        # ✅ CHANGÉ: Nouvelle base de données
        self.db = mysql.connector.connect(
            host="localhost", 
            user="root", 
            password="2929", 
            database="instagram_ml_real"  # ← NOUVELLE DB
        )
        self.cursor = self.db.cursor()
        self.user_post_matrix, self.users, self.posts_df = self.load_real_data()
        self.rf_model = self.train_model()
        self.image_cache = {}
        print(f"✅ Connecté à instagram_ml_real: {len(self.users)} users, {len(self.posts_df)} posts")
    
    def load_real_data(self):
        """Charge posts AVEC images + captions - ADAPTÉ nouvelle DB"""
        self.cursor.execute("""
            SELECT 
                u.username, 
                p.post_name, 
                p.image_path, 
                p.caption,
                COUNT(i.interaction_id) as interaction_count
            FROM interactions i
            JOIN users u ON i.user_id = u.user_id
            JOIN posts p ON i.post_id = p.post_id
            GROUP BY u.username, p.post_name, p.image_path, p.caption
        """)
        
        data = self.cursor.fetchall()
        if not data:
            raise ValueError("❌ Aucune donnée trouvée! Vérifiez la DB instagram_ml_real")
        
        users = sorted(set(row[0] for row in data))
        post_names = sorted(set(row[1] for row in data))
        
        # Matrice ML (user x post)
        matrix = pd.DataFrame(0, index=users, columns=post_names, dtype=np.int8)
        
        # DataFrame posts COMPLET (image_path + caption)
        posts_dict = {}
        for row in data:
            post_name = row[1]
            posts_dict[post_name] = {
                'image_path': row[2] or f'images/{post_name}.jpg',
                'caption': row[3] or f'Post {post_name} - Super content!'
            }
        posts_df = pd.DataFrame(posts_dict).T
        
        # Remplir matrice interactions
        for username, post_name, image_path, caption, count in data:
            matrix.loc[username, post_name] = min(1, count)
        
        return matrix, users, posts_df
    
    def load_image(self, post_name, size=(150, 150)):
        """Charge image avec fallback"""
        if post_name in self.image_cache:
            return self.image_cache[post_name]
        
        try:
            if post_name not in self.posts_df.index:
                return None
            
            post_info = self.posts_df.loc[post_name]
            image_path = post_info['image_path']
            
            # Fallback si image n'existe pas
            fallback_paths = [
                image_path,
                f'images/{post_name}.jpg',
                'images/default.jpg'
            ]
            
            for path in fallback_paths:
                if os.path.exists(path):
                    img = Image.open(path)
                    img = img.resize(size, Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.image_cache[post_name] = photo
                    return photo
            
            # Image par défaut si rien trouvé
            return self.create_default_image(size, post_name)
            
        except Exception as e:
            print(f"⚠️ Image {post_name} non chargée: {e}")
            return self.create_default_image(size, post_name)
    
    def create_default_image(self, size, post_name):
        """Crée image placeholder"""
        img = Image.new('RGB', size, color='#e0e0e0')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), post_name[:10], fill='black')
        photo = ImageTk.PhotoImage(img)
        self.image_cache[post_name] = photo
        return photo
    
    def train_model(self):
        """Modèle RandomForest adapté"""
        X, y = [], []
        for user in self.users:
            for post in self.posts_df.index:
                post_fans = self.user_post_matrix[post][self.user_post_matrix[post] > 0]
                sim_to_fans = 0
                if len(post_fans) > 0:
                    user_vec = self.user_post_matrix.loc[user].values.reshape(1, -1)
                    sim_to_fans = np.mean([
                        cosine_similarity(user_vec, 
                                        self.user_post_matrix.loc[fan].values.reshape(1, -1))[0][0] 
                        for fan in post_fans.index if fan in self.users
                    ])
                X.append([len(post_fans), sim_to_fans])
                y.append(self.user_post_matrix.loc[user, post] if post in self.user_post_matrix.columns else 0)
        
        if len(X) > 10:  # Minimum pour entraîner
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            return model
        return None
    
    def get_real_recommendations(self, target_user, n=5):
        """Recommandations ML"""
        if target_user not in self.users:
            return [], []
        
        # SVD pour similarité
        n_components = min(10, self.user_post_matrix.shape[1]-1, len(self.users)-1)
        svd = TruncatedSVD(n_components=n_components)
        matrix_reduced = svd.fit_transform(self.user_post_matrix.fillna(0))
        
        similarity_matrix = cosine_similarity(matrix_reduced)
        target_idx = self.users.index(target_user)
        sim_scores = dict(zip(self.users, similarity_matrix[target_idx]))
        
        # Top 3 similaires
        top_similar = sorted(
            [(u, score) for u, score in sim_scores.items() if u != target_user],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        # Posts recommandés
        target_likes = set(self.user_post_matrix.loc[target_user][self.user_post_matrix.loc[target_user] > 0].index)
        recommendations = []
        
        for sim_user, score in top_similar:
            if sim_user in self.user_post_matrix.index:
                sim_likes = set(self.user_post_matrix.loc[sim_user][self.user_post_matrix.loc[sim_user] > 0].index)
                for post in (sim_likes - target_likes):
                    if post in self.posts_df.index:
                        recommendations.append((post, score))
        
        top_posts = sorted(recommendations, key=lambda x: x[1], reverse=True)[:n]
        return top_posts, [user for user, _ in top_similar]

    def __del__(self):
        self.cursor.close()
        self.db.close()

# 🖼️ INTERFACE ADAPTÉE
class InstagramRealApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📱 Instagram ML Recommender - instagram_ml_real")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f2f5')
        
        try:
            self.recommender = InstagramRecommenderReal()
            self.setup_real_ui()
        except Exception as e:
            messagebox.showerror("Erreur DB", f"Connexion échouée:\n{str(e)}\n\nVérifiez:\n• DB 'instagram_ml_real' existe\n• MySQL tourne\n• Données insérées")
            root.destroy()
    
    def setup_real_ui(self):
        # Header
        header = tk.Frame(self.root, bg='#1877f2', height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        title = tk.Label(header, text="📱 Instagram ML Recommender", 
                        font=('Arial', 24, 'bold'), bg='#1877f2', fg='white')
        title.pack(pady=20)
        
        # Contenu
        content = tk.Frame(self.root, bg='#f0f2f5')
        content.pack(fill='both', expand=True, padx=20, pady=20)
        
        # GAUCHE: Sélecteur
        left_frame = tk.Frame(content, bg='#f0f2f5', width=300)
        left_frame.pack(side='left', fill='y', padx=(0,20))
        left_frame.pack_propagate(False)
        
        tk.Label(left_frame, text="👤 Choisir utilisateur", 
                font=('Arial', 14, 'bold'), bg='#f0f2f5').pack(pady=20)
        
        self.user_var = tk.StringVar()
        user_combo = ttk.Combobox(left_frame, textvariable=self.user_var,
                                 values=self.recommender.users, state='readonly', width=20)
        user_combo.pack(pady=10)
        if self.recommender.users:
            user_combo.set(self.recommender.users[0])
        
        tk.Button(left_frame, text="🔥 Recommandations ML", command=self.get_real_recs,
                 bg='#ff4b5c', fg='white', font=('Arial', 14, 'bold'),
                 cursor='hand2', width=18).pack(pady=20)
        
        self.stats_label = tk.Label(left_frame, text="", bg='#f0f2f5', 
                                   font=('Arial', 11), fg='#65676b')
        self.stats_label.pack(pady=20)
        
        # DROITE: Galerie
        right_frame = tk.Frame(content, bg='#f0f2f5')
        right_frame.pack(side='right', fill='both', expand=True)
        
        gallery_frame = tk.LabelFrame(right_frame, text="🎯 Posts Recommandés", 
                                     font=('Arial', 16, 'bold'), bg='#f0f2f5', fg='#1877f2')
        gallery_frame.pack(fill='both', expand=True, pady=(0,20))
        
        # Canvas scrollable
        canvas = tk.Canvas(gallery_frame, bg='white')
        scrollbar = ttk.Scrollbar(gallery_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg='white')
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        self.post_frames = []
    
    def get_real_recs(self):
        if not self.recommender.users:
            messagebox.showwarning("Aucune donnée", "Aucun utilisateur trouvé dans la DB!")
            return
        threading.Thread(target=self.update_real_recs, daemon=True).start()
    
    def update_real_recs(self):
        target_user = self.user_var.get()
        try:
            top_posts, similar_users = self.recommender.get_real_recommendations(target_user)
            self.root.after(0, self.display_real_posts, target_user, top_posts, similar_users)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Erreur ML", str(e)))
    
    def display_real_posts(self, target_user, top_posts, similar_users):
        # Nettoyer
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.post_frames = []
        
        # Stats
        stats_text = f"👥 Similaires: {', '.join(similar_users[:3])}"
        if top_posts:
            stats_text += f" | 📸 {len(top_posts)} posts"
        else:
            stats_text += " | 😴 Aucune reco trouvée"
        self.stats_label.config(text=stats_text)
        
        # Cards Instagram
        if top_posts:
            for i, (post_name, score) in enumerate(top_posts, 1):
                frame = tk.Frame(self.scrollable_frame, bg='white', relief='raised', bd=2)
                frame.pack(fill='x', padx=15, pady=8)
                
                # Image
                img = self.recommender.load_image(post_name)
                img_label = tk.Label(frame, image=img or None, bg='white', width=150, height=150)
                img_label.image = img  # Référence
                img_label.pack(side='left', padx=15, pady=15)
                
                # Infos
                info_frame = tk.Frame(frame, bg='white')
                info_frame.pack(side='right', fill='both', expand=True, padx=15, pady=15)
                
                tk.Label(info_frame, text=f"{i}. {post_name}", 
                        font=('Arial', 12, 'bold'), bg='white', fg='#262626').pack(anchor='w')
                
                caption = self.recommender.posts_df.loc[post_name, 'caption'] if post_name in self.recommender.posts_df.index else "Caption non disponible"
                tk.Label(info_frame, text=caption[:120] + '...' if len(caption) > 120 else caption, 
                        font=('Arial', 10), bg='white', fg='#65676b', 
                        wraplength=400, justify='left').pack(anchor='w')
                
                tk.Label(info_frame, text=f"🤖 Score ML: {score:.3f}", 
                        font=('Arial', 10, 'bold'), bg='white', fg='#1877f2').pack(anchor='e')
                
                self.post_frames.append(frame)
        else:
            no_rec_label = tk.Label(self.scrollable_frame, text="😴 Aucune recommandation trouvée\nEssayez un autre utilisateur", 
                                  font=('Arial', 16), bg='white', fg='#999')
            no_rec_label.pack(expand=True)

# 🚀 LANCEMENT
if __name__ == "__main__":
    root = tk.Tk()
    app = InstagramRealApp(root)
    root.mainloop()
