import tkinter as tk
from tkinter import ttk, messagebox
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import warnings
warnings.filterwarnings('ignore')

# 🔗 VOTRE CODE ML (fonctions clés)
class InstagramRecommender:
    def __init__(self):
        self.db = mysql.connector.connect(
            host="localhost", user="root", password="2929", database="instagrame_data"
        )
        self.cursor = self.db.cursor()
        self.user_post_matrix, self.users, self.posts = self.load_data()
        self.rf_model = self.train_model()
        print("✅ Système ML chargé!")
    
    def load_data(self):
        self.cursor.execute("""
            SELECT u.username, p.post_name, COUNT(*) as interaction_count
            FROM interactions i
            JOIN users u ON i.user_id = u.user_id
            JOIN posts p ON i.post_id = p.post_id
            GROUP BY u.username, p.post_name
        """)
        data = self.cursor.fetchall()
        users = sorted(set(row[0] for row in data))
        posts = sorted(set(row[1] for row in data))
        
        matrix = pd.DataFrame(0, index=users, columns=posts, dtype=np.int8)
        for username, post_name, count in data:
            matrix.loc[username, post_name] = min(1, count)
        return matrix, users, posts
    
    def train_model(self):
        X, y = [], []
        for user in self.users:
            for post in self.posts:
                post_fans = self.user_post_matrix[post][self.user_post_matrix[post] > 0]
                sim_to_fans = 0
                if len(post_fans) > 0:
                    user_vec = self.user_post_matrix.loc[user].values.reshape(1,-1)
                    sim_to_fans = np.mean([
                        cosine_similarity(user_vec, 
                                        self.user_post_matrix.loc[fan].values.reshape(1,-1))[0][0] 
                        for fan in post_fans.index
                    ])
                X.append([len(post_fans), sim_to_fans])
                y.append(self.user_post_matrix.loc[user, post])
        
        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    def get_recommendations(self, target_user, n=5):
        svd = TruncatedSVD(n_components=min(50, self.user_post_matrix.shape[1]-1))
        matrix_reduced = svd.fit_transform(self.user_post_matrix)
        similarity_matrix = cosine_similarity(matrix_reduced)
        
        target_idx = self.users.index(target_user)
        sim_scores = dict(zip(self.users, similarity_matrix[target_idx]))
        
        top_similar = sorted(
            [(u, score) for u, score in sim_scores.items() if u != target_user],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        target_likes = set(self.user_post_matrix.loc[target_user][self.user_post_matrix.loc[target_user] > 0].index)
        candidate_posts = []
        
        for sim_user, score in top_similar:
            sim_likes = set(self.user_post_matrix.loc[sim_user][self.user_post_matrix.loc[sim_user] > 0].index)
            for post in sim_likes - target_likes:
                candidate_posts.append((post, score))
        
        recommendations = sorted(candidate_posts, key=lambda x: x[1], reverse=True)
        return [post for post, _ in recommendations[:n]], [user for user, _ in top_similar]

# 🎨 INTERFACE GRAPHIQUE
class RecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Instagram ML Recommender")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f2f5')
        
        self.recommender = InstagramRecommender()
        self.setup_ui()
    
    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="🎯 Système de Recommandation Instagram ML", 
                        font=('Arial', 20, 'bold'), bg='#f0f2f5', fg='#1877f2')
        title.pack(pady=20)
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#f0f2f5')
        main_frame.pack(fill='both', expand=True, padx=20)
        
        # Left: Sélection utilisateur
        left_frame = tk.Frame(main_frame, bg='#f0f2f5')
        left_frame.pack(side='left', fill='y', padx=(0,20))
        
        tk.Label(left_frame, text="👤 Choisir un utilisateur:", 
                font=('Arial', 12, 'bold'), bg='#f0f2f5').pack(pady=10)
        
        self.user_var = tk.StringVar()
        users_combo = ttk.Combobox(left_frame, textvariable=self.user_var, 
                                  values=self.recommender.users, state='readonly')
        users_combo.pack(pady=10, ipadx=50)
        users_combo.set(self.recommender.users[0])
        
        tk.Button(left_frame, text="🔥 Obtenir Recommandations", 
                 command=self.get_recs, bg='#1877f2', fg='white',
                 font=('Arial', 12, 'bold'), cursor='hand2').pack(pady=20)
        
        # Stats
        self.stats_label = tk.Label(left_frame, text="", bg='#f0f2f5', 
                                   font=('Arial', 10), fg='gray')
        self.stats_label.pack(pady=10)
        
        # Right: Résultats
        right_frame = tk.Frame(main_frame, bg='#f0f2f5')
        right_frame.pack(side='right', fill='both', expand=True)
        
        # Recommandations
        rec_frame = tk.LabelFrame(right_frame, text="🎯 Recommandations ML", 
                                 font=('Arial', 14, 'bold'), bg='#f0f2f5', fg='#1877f2')
        rec_frame.pack(fill='both', expand=True, pady=(0,20))
        
        self.rec_listbox = tk.Listbox(rec_frame, font=('Arial', 11), 
                                     bg='white', selectbackground='#1877f2')
        self.rec_listbox.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Graphique
        graph_frame = tk.LabelFrame(right_frame, text="📊 Analyse ML", 
                                   font=('Arial', 14, 'bold'), bg='#f0f2f5', fg='#1877f2')
        graph_frame.pack(fill='x')
        
        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def get_recs(self):
        target_user = self.user_var.get()
        if target_user not in self.recommender.users:
            messagebox.showerror("Erreur", "Utilisateur non trouvé!")
            return
        
        # Thread pour éviter blocage UI
        threading.Thread(target=self.update_recommendations, args=(target_user,), daemon=True).start()
    
    def update_recommendations(self, target_user):
        try:
            recs, similar_users = self.recommender.get_recommendations(target_user)
            
            # Mise à jour UI (thread-safe)
            self.root.after(0, self.display_results, target_user, recs, similar_users)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Erreur", str(e)))
    
    def display_results(self, target_user, recs, similar_users):
        # Clear listbox
        self.rec_listbox.delete(0, tk.END)
        
        # Afficher recommandations
        for i, post in enumerate(recs, 1):
            self.rec_listbox.insert(tk.END, f"{i}. 🎬 {post}")
        
        # Stats
        self.stats_label.config(
            text=f"👥 Similaires: {', '.join(similar_users)} | 🎯 {len(recs)} recs pour {target_user}"
        )
        
        # Graphique similarités
        self.ax.clear()
        scores = [1.0/len(similar_users) * (i+1) for i in range(len(similar_users))]
        self.ax.barh(similar_users, scores, color='#1877f2', alpha=0.8)
        self.ax.set_title(f'Utilisateurs similaires à {target_user}', fontweight='bold')
        self.ax.set_xlabel('Score de similarité')
        self.fig.tight_layout()
        self.canvas.draw()
        
        print(f"✅ Recommandations pour {target_user}: {recs}")
    
    def __del__(self):
        self.recommender.cursor.close()
        self.recommender.db.close()

# 🚀 LANCEMENT APPLICATION
if __name__ == "__main__":
    root = tk.Tk()
    app = RecommenderApp(root)
    root.mainloop()
