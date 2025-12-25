import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import threading


class InstagramRecommenderReal:
    def __init__(self):
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

        matrix = pd.DataFrame(0, index=users, columns=post_names, dtype=np.int8)

        posts_dict = {}
        for row in data:
            post_name = row[1]
            posts_dict[post_name] = {
                "image_path": row[2] or f"images/{post_name}.jpg",
                "caption": row[3] or f"Post {post_name} - contenu indisponible"
            }
        posts_df = pd.DataFrame(posts_dict).T

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
        """Crée une image avec tes couleurs."""
        img = Image.new("RGB", size, color="#1D4241")
        draw = ImageDraw.Draw(img)
        text = post_name[:10]
        draw.text((10, 10), text, fill="#F9EEE7")
        photo = ImageTk.PhotoImage(img)
        self.image_cache[post_name] = photo
        return photo

    def train_model(self):
        """Entraîne un RandomForest pour apprendre like / pas like."""
        X, y = [], []
        for user in self.users:
            for post in self.posts_df.index:
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
        """Renvoie (posts_recommandés, utilisateurs_similaires) SANS SVD."""
        if target_user not in self.users:
            return [], []

        
        matrix_for_sim = self.user_post_matrix.fillna(0)
        similarity_matrix = cosine_similarity(matrix_for_sim)
        target_idx = self.users.index(target_user)
        sim_scores = dict(zip(self.users, similarity_matrix[target_idx]))

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
                        if post not in recommendations or score > recommendations[post]:
                            recommendations[post] = score

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
        self.root.configure(bg="#1D4241") 

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
        header = tk.Frame(self.root, bg="#123332", height=80)
        header.pack(fill="x")
        header.pack_propagate(False)

        title = tk.Label(
            header,
            text="📱 Instagram ML Recommender",
            font=("Arial", 24, "bold"),
            bg="#123332",  
            fg="#F9EEE7",  
        )
        title.pack(pady=10)

        subtitle = tk.Label(
            header,
            text="Recommandations personnalisées à partir de vos interactions",
            font=("Arial", 11),
            bg="#123332",  
            fg="#F9EEE7",  
        )
        subtitle.pack()

        content = tk.Frame(self.root, bg="#1D4241")  
        content.pack(fill="both", expand=True, padx=20, pady=20)
       
        left_frame = tk.Frame(content, bg="#FFD9BE", width=260)  # ✅ #FFD9BE
        left_frame.pack(side="left", fill="y", padx=(0, 20))
        left_frame.pack_propagate(False)

        tk.Label(
            left_frame,
            text="👤 Entrer votre username",
            font=("Arial", 14, "bold"),
            bg="#FFD9BE",  
            fg="#123332",  
        ).pack(pady=(10, 5))

        self.user_entry = tk.Entry(
            left_frame,
            font=("Arial", 12),
            width=22,
            bg="#1D4241",  
            fg="#F9EEE7",  
            insertbackground="#F9EEE7",  
            highlightbackground="#123332",  
            selectbackground="#123332"  
        )
        self.user_entry.pack(pady=5)

        tk.Button(
            left_frame,
            text=" Recommander ",
            command=self.get_real_recs,
            bg="#123332",  
            fg="#F9EEE7",  
            font=("Arial", 13, "bold"),
            cursor="hand2",
            width=20,
        ).pack(pady=15)

        self.stats_label = tk.Label(
            left_frame,
            text="",
            bg="#FFD9BE",  
            font=("Arial", 10),
            fg="#123332",  
            justify="left",
        )
        self.stats_label.pack(pady=10)
        
        right_frame = tk.Frame(content, bg="#1D4241")  
        right_frame.pack(side="right", fill="both", expand=True)

        gallery_frame = tk.LabelFrame(
            right_frame,
            text="🎯 Posts recommandés",
            font=("Arial", 16, "bold"),
            bg="#FFD9BE",  
            fg="#123332", 
        )
        gallery_frame.pack(fill="both", expand=True)
       
        self.canvas = tk.Canvas(gallery_frame, bg="#1D4241", highlightthickness=0)  
        scrollbar = ttk.Scrollbar(gallery_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#1D4241")  

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.canvas.bind(
            "<Configure>",
            self._on_canvas_configure
        )
        self.scrollable_frame.bind(
            "<Configure>",
            self._on_frame_configure
        )
        
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self._scroll_canvas(-1))
        self.canvas.bind("<Button-5>", lambda e: self._scroll_canvas(1))

    def _on_canvas_configure(self, event):
        """Mise à jour de la scrollregion du canvas."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_frame_configure(self, event):
        """Mise à jour quand le frame change de taille."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        """Scroll avec molette."""
        self.canvas.yview_scroll(-int(event.delta/120), "units")

    def _scroll_canvas(self, steps):
        """Scroll pour Linux."""
        self.canvas.yview_scroll(steps, "units")

    def get_real_recs(self):
        target_user = self.user_entry.get().strip()

        if not target_user:
            messagebox.showwarning(
                "Nom manquant",
                "Veuillez entrer votre username."
            )
            return

        threading.Thread(
            target=self.update_real_recs,
            args=(target_user,),
            daemon=True
        ).start()

    def update_real_recs(self, target_user):
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
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
       
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
                bg="#1D4241",  
                fg="#F9EEE7",  
            ).pack(pady=40)
            return
       
        for i, (post_name, score) in enumerate(top_posts, start=1):
            card = tk.Frame(self.scrollable_frame, bg="#EF9C82", relief="solid", bd=1)  # ✅ #EF9C82
            card.pack(fill="x", padx=10, pady=10)

            # Image
            img = self.recommender.load_image(post_name)
            img_label = tk.Label(card, image=img, bg="#1D4241", width=160, height=160)  
            img_label.image = img
            img_label.pack(side="left", padx=10, pady=10)

            # Infos
            info = tk.Frame(card, bg="#EF9C82")  
            info.pack(side="right", fill="both", expand=True, padx=10, pady=10)

            tk.Label(
                info,
                text=f"{i}. {post_name}",
                font=("Arial", 12, "bold"),
                bg="#EF9C82",  
                fg="#123332",  
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
                bg="#EF9C82",  
                fg="#1D4241",  
                wraplength=500,
                justify="left",
            ).pack(anchor="w", pady=(5, 5))

            tk.Label(
                info,
                text=f"🤖 Score ML : {score:.3f}",
                font=("Arial", 10, "bold"),
                bg="#EF9C82",  
                fg="#123332", 
            ).pack(anchor="e")
        
        self.canvas.update_idletasks()
        self._on_frame_configure(None)


if __name__ == "__main__":
    root = tk.Tk()
    app = InstagramRealApp(root)
    root.mainloop()
