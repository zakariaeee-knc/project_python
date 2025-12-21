import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import threading
import warnings

warnings.filterwarnings('ignore')


class InstagramRecommenderReal:
    def __init__(self):
        # Connexion DB
        self.db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="2929",
            database="instagram_ml_real"
        )
        self.cursor = self.db.cursor()
        self.user_post_matrix, self.users, self.posts_df = self.load_real_data()
        self.rf_model = self.train_model()
        self.image_cache = {}

    def load_real_data(self):
        """Charge les données réelles : users, posts, interactions."""
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
            raise ValueError("Aucune donnée trouvée dans la base instagram_ml_real.")

        users = sorted(set(row[0] for row in data))
        post_names = sorted(set(row[1] for row in data))

        # Matrice (users x posts) binaire
        matrix = pd.DataFrame(0, index=users, columns=post_names, dtype=np.int8)

        # DataFrame des posts (image_path + caption)
        posts_dict = {}
        for row in data:
            post_name = row[1]
            posts_dict[post_name] = {
                "image_path": row[2] or f"images/{post_name}.jpg",
                "caption": row[3] or f"Post {post_name} - contenu indisponible"
            }
        posts_df = pd.DataFrame(posts_dict).T

        # Remplir la matrice d’interactions
        for username, post_name, image_path, caption, count in data:
            if username in matrix.index and post_name in matrix.columns:
                matrix.loc[username, post_name] = 1

        return matrix, users, posts_df

    def load_image(self, post_name, size=(150, 150)):
        """Charge une image (avec cache et fallback)."""
        if post_name in self.image_cache:
            return self.image_cache[post_name]

        if post_name not in self.posts_df.index:
            return self.create_default_image(size, post_name)

        post_info = self.posts_df.loc[post_name]
        image_path = post_info["image_path"]

        fallback_paths = [
            image_path,
            f"images/{post_name}.jpg",
            "images/default.jpg"
        ]

        for path in fallback_paths:
            if path and os.path.exists(path):
                try:
                    img = Image.open(path)
                    img = img.resize(size, Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.image_cache[post_name] = photo
                    return photo
                except Exception:
                    continue

        return self.create_default_image(size, post_name)

    def create_default_image(self, size, post_name):
        """Crée une image grise par défaut."""
        img = Image.new("RGB", size, color="#e0e0e0")
        draw = ImageDraw.Draw(img)
        text = post_name[:10]
        draw.text((10, 10), text, fill="black")
        photo = ImageTk.PhotoImage(img)
        self.image_cache[post_name] = photo
        return photo

    def train_model(self):
        """Entraîne un RandomForest pour apprendre like / pas like."""
        X, y = [], []
        for user in self.users:
            for post in self.posts_df.index:
                # fans du post
                post_fans = self.user_post_matrix[post][self.user_post_matrix[post] > 0]
                sim_to_fans = 0.0
                if len(post_fans) > 0:
                    user_vec = self.user_post_matrix.loc[user].values.reshape(1, -1)
                    sims = []
                    for fan in post_fans.index:
                        if fan in self.users:
                            fan_vec = self.user_post_matrix.loc[fan].values.reshape(1, -1)
                            s = cosine_similarity(user_vec, fan_vec)[0][0]
                            sims.append(s)
                    if sims:
                        sim_to_fans = float(np.mean(sims))

                X.append([len(post_fans), sim_to_fans])
                y.append(int(self.user_post_matrix.loc[user, post]))

        if len(X) > 10:
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            return model
        return None

    def get_real_recommendations(self, target_user, n=5):
        """Renvoie (posts_recommandés, utilisateurs_similaires)."""
        if target_user not in self.users:
            return [], []

        # SVD pour trouver les utilisateurs similaires
        n_components = min(10, self.user_post_matrix.shape[1] - 1, len(self.users) - 1)
        svd = TruncatedSVD(n_components=n_components)
        matrix_reduced = svd.fit_transform(self.user_post_matrix.fillna(0))

        similarity_matrix = cosine_similarity(matrix_reduced)
        target_idx = self.users.index(target_user)
        sim_scores = dict(zip(self.users, similarity_matrix[target_idx]))

        # Top 3 utilisateurs similaires
        top_similar = sorted(
            [(u, s) for u, s in sim_scores.items() if u != target_user],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        target_likes = set(
            self.user_post_matrix.loc[target_user][
                self.user_post_matrix.loc[target_user] > 0
            ].index
        )

        recommendations = {}
        for sim_user, score in top_similar:
            if sim_user in self.user_post_matrix.index:
                sim_likes = set(
                    self.user_post_matrix.loc[sim_user][
                        self.user_post_matrix.loc[sim_user] > 0
                    ].index
                )
                for post in sim_likes - target_likes:
                    if post in self.posts_df.index:
                        # évite les répétitions : on garde le meilleur score
                        if post not in recommendations or score > recommendations[post]:
                            recommendations[post] = score

        # tri par score
        sorted_posts = sorted(
            recommendations.items(), key=lambda x: x[1], reverse=True
        )[:n]

        similar_users = [u for u, _ in top_similar]
        return sorted_posts, similar_users

    def __del__(self):
        try:
            self.cursor.close()
            self.db.close()
        except Exception:
            pass


class InstagramRealApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📱 Instagram ML Recommender")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f2f5")

        try:
            self.recommender = InstagramRecommenderReal()
            self.setup_ui()
        except Exception as e:
            messagebox.showerror(
                "Erreur DB",
                f"Connexion échouée :\n{e}\n\nVérifiez la base 'instagram_ml_real' et MySQL."
            )
            root.destroy()

    def setup_ui(self):
        # HEADER simple (couleur unie, pas de gradient)
        header = tk.Frame(self.root, bg="#1877f2", height=80)
        header.pack(fill="x")
        header.pack_propagate(False)

        title = tk.Label(
            header,
            text="📱 Instagram ML Recommender",
            font=("Arial", 24, "bold"),
            bg="#1877f2",
            fg="white",
        )
        title.pack(pady=10)

        subtitle = tk.Label(
            header,
            text="Recommandations personnalisées à partir de vos interactions",
            font=("Arial", 11),
            bg="#1877f2",
            fg="white",
        )
        subtitle.pack()

        # CONTENU PRINCIPAL
        content = tk.Frame(self.root, bg="#f0f2f5")
        content.pack(fill="both", expand=True, padx=20, pady=20)

        # PANEL GAUCHE : sélection utilisateur
        left_frame = tk.Frame(content, bg="#f0f2f5", width=260)
        left_frame.pack(side="left", fill="y", padx=(0, 20))
        left_frame.pack_propagate(False)

        tk.Label(
            left_frame,
            text="👤 Choisir un utilisateur",
            font=("Arial", 14, "bold"),
            bg="#f0f2f5",
        ).pack(pady=(10, 10))

        self.user_var = tk.StringVar()
        user_combo = ttk.Combobox(
            left_frame,
            textvariable=self.user_var,
            values=self.recommender.users,
            state="readonly",
            width=20,
        )
        user_combo.pack(pady=5)
        if self.recommender.users:
            user_combo.set(self.recommender.users[0])

        tk.Button(
            left_frame,
            text="🔥 Recommandations ML",
            command=self.get_real_recs,
            bg="#ff4b5c",
            fg="white",
            font=("Arial", 13, "bold"),
            cursor="hand2",
            width=20,
        ).pack(pady=15)

        self.stats_label = tk.Label(
            left_frame,
            text="",
            bg="#f0f2f5",
            font=("Arial", 10),
            fg="#555555",
            justify="left",
        )
        self.stats_label.pack(pady=10)

        # PANEL DROIT : galerie scrollable
        right_frame = tk.Frame(content, bg="#f0f2f5")
        right_frame.pack(side="right", fill="both", expand=True)

        gallery_frame = tk.LabelFrame(
            right_frame,
            text="🎯 Posts recommandés",
            font=("Arial", 16, "bold"),
            bg="#f0f2f5",
            fg="#1877f2",
        )
        gallery_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(gallery_frame, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            gallery_frame, orient="vertical", command=canvas.yview
        )
        self.scrollable_frame = tk.Frame(canvas, bg="white")

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        canvas.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

    def get_real_recs(self):
        if not self.recommender.users:
            messagebox.showwarning(
                "Aucune donnée",
                "Aucun utilisateur trouvé dans la base de données.",
            )
            return
        threading.Thread(target=self.update_real_recs, daemon=True).start()

    def update_real_recs(self):
        target_user = self.user_var.get()
        try:
            top_posts, similar_users = self.recommender.get_real_recommendations(
                target_user
            )
            self.root.after(
                0, self.display_real_posts, target_user, top_posts, similar_users
            )
        except Exception as e:
            self.root.after(
                0, lambda: messagebox.showerror("Erreur ML", str(e))
            )

    def display_real_posts(self, target_user, top_posts, similar_users):
        # nettoyage
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # stats
        stats_text = f"Utilisateur : {target_user}"
        if similar_users:
            stats_text += f"\nSimilaires : {', '.join(similar_users[:3])}"
        if top_posts:
            scores = [s for _, s in top_posts]
            stats_text += f"\nNombre de posts : {len(top_posts)} | Score moyen : {np.mean(scores):.2f}"
        else:
            stats_text += "\nAucune recommandation trouvée."
        self.stats_label.config(text=stats_text)

        if not top_posts:
            tk.Label(
                self.scrollable_frame,
                text="😴 Aucune recommandation pour cet utilisateur",
                font=("Arial", 14),
                bg="white",
                fg="#999999",
            ).pack(pady=40)
            return

        # affichage des cards
        for i, (post_name, score) in enumerate(top_posts, start=1):
            card = tk.Frame(self.scrollable_frame, bg="white", relief="solid", bd=1)
            card.pack(fill="x", padx=10, pady=10)

            # image
            img = self.recommender.load_image(post_name)
            img_label = tk.Label(card, image=img, bg="white", width=160, height=160)
            img_label.image = img  # éviter le garbage collector
            img_label.pack(side="left", padx=10, pady=10)

            # infos
            info = tk.Frame(card, bg="white")
            info.pack(side="right", fill="both", expand=True, padx=10, pady=10)

            tk.Label(
                info,
                text=f"{i}. {post_name}",
                font=("Arial", 12, "bold"),
                bg="white",
                fg="#222222",
            ).pack(anchor="w")

            caption = (
                self.recommender.posts_df.loc[post_name, "caption"]
                if post_name in self.recommender.posts_df.index
                else "Caption non disponible"
            )
            if len(caption) > 140:
                caption = caption[:140] + "..."
            tk.Label(
                info,
                text=caption,
                font=("Arial", 10),
                bg="white",
                fg="#555555",
                wraplength=500,
                justify="left",
            ).pack(anchor="w", pady=(5, 5))

            tk.Label(
                info,
                text=f"🤖 Score ML : {score:.3f}",
                font=("Arial", 10, "bold"),
                bg="white",
                fg="#1877f2",
            ).pack(anchor="e")


if __name__ == "__main__":
    root = tk.Tk()
    app = InstagramRealApp(root)
    root.mainloop()
